//! Complex NIfTI I/O.
//!
//! GIFT stores complex data as TWO ordinary NIfTI files per volume (image-domain
//! reconstructed data, not raw k-space), distinguished by a prefix around an underscore:
//! R_/I_ for real&imaginary, Mag_/Phase_ for magnitude&phase.
//! Mirrors icatb_loadData.m:79-85.

use std::path::{Path, PathBuf};

use nifti::{IntoNdArray, NiftiObject, NiftiVolume, ReaderOptions};
use num_complex::Complex64;

#[derive(Debug, Clone, Copy)]
pub enum ComplexType {
    RealImag,
    MagPhase,
}

fn read_volume(path: &Path) -> Result<(Vec<f64>, [usize; 4]), String> {
    let obj = ReaderOptions::new()
        .read_file(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let vol = obj.volume();
    let d = vol.dim();
    let dims = [
        *d.first().unwrap_or(&1) as usize,
        *d.get(1).unwrap_or(&1) as usize,
        *d.get(2).unwrap_or(&1) as usize,
        *d.get(3).unwrap_or(&1) as usize,
    ];
    let arr = vol
        .into_ndarray::<f64>()
        .map_err(|e| format!("cannot decode {}: {e}", path.display()))?;
    // `into_ndarray` hands back the volume in Fortran (x-fastest) memory order, matching the
    // raw on-disk NIfTI layout. Convert to standard (C, last-axis-fastest) layout before
    // flattening so the returned Vec enumerates voxels in the same order callers (and the
    // fixtures) expect: index = i*(ny*nz) + j*nz + k for dims [nx, ny, nz].
    let data: Vec<f64> = arr.as_standard_layout().into_owned().into_raw_vec_and_offset().0;
    Ok((data, dims))
}

/// Assemble one complex volume from GIFT's two-file representation.
pub fn read_complex(
    first: &Path,
    second: &Path,
    kind: ComplexType,
) -> Result<(Vec<Complex64>, [usize; 4]), String> {
    let (a, dims_a) = read_volume(first)?;
    let (b, dims_b) = read_volume(second)?;
    if dims_a != dims_b {
        return Err(format!("shape mismatch: {dims_a:?} vs {dims_b:?}"));
    }
    let data = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| match kind {
            ComplexType::RealImag => Complex64::new(x, y),
            ComplexType::MagPhase => Complex64::new(x * y.cos(), x * y.sin()),
        })
        .collect();
    Ok((data, dims_a))
}

/// Split a complex volume back into GIFT's two-file representation.
pub fn write_complex(
    data: &[Complex64],
    dims: [usize; 3],
    first: &Path,
    second: &Path,
    kind: ComplexType,
) -> Result<(), String> {
    use nifti::writer::WriterOptions;

    let n = dims[0] * dims[1] * dims[2];
    if data.len() != n {
        return Err(format!("data has {} values, dims imply {n}", data.len()));
    }
    let (a, b): (Vec<f64>, Vec<f64>) = data
        .iter()
        .map(|z| match kind {
            ComplexType::RealImag => (z.re, z.im),
            ComplexType::MagPhase => (z.norm(), z.arg()),
        })
        .unzip();

    for (path, vals) in [(first, a), (second, b)] {
        let arr = ndarray::Array3::from_shape_vec((dims[0], dims[1], dims[2]), vals)
            .map_err(|e| format!("bad shape for {}: {e}", path.display()))?;
        WriterOptions::new(path)
            .write_nifti(&arr)
            .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    }
    Ok(())
}

/// Derive GIFT's two filenames from a base name (a prefix around an underscore).
pub fn complex_file_pair(
    path: &Path,
    naming: (&str, &str),
) -> Result<(PathBuf, PathBuf), String> {
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| format!("not a file path: {}", path.display()))?;
    if !name.contains('_') {
        return Err(format!(
            "complex file names must contain an underscore (GIFT convention): {name}"
        ));
    }
    let dir = path.parent().unwrap_or(Path::new("."));
    Ok((
        dir.join(format!("{}{}", naming.0, name)),
        dir.join(format!("{}{}", naming.1, name)),
    ))
}
