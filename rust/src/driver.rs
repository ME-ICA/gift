//! File-level driver: complex NIfTI in, complex component maps out.
//!
//! `pipeline::run_complex_ica` is array-in/array-out and returns maps over the masked
//! voxels only. This is the thin layer that makes the crate usable on real data: it loads
//! GIFT's two-file complex volumes, flattens them, runs the pipeline, expands the maps back
//! to full volume shape, and writes them out as complex NIfTI pairs.
//!
//! Component NUMBERING IS NOT STABLE ACROSS PORTS OR RUNS. ICA fixes neither the order nor
//! the phase of its components, and no port canonicalizes them, so `component_001` here and
//! `component_001` from the Python driver are generally DIFFERENT sources - even for the
//! deterministic estimator, where the two agree to ~4e-14 once matched up. Match components
//! by correlation, never by filename.
//!
//! Mirrors `python/src/gift/driver.py`; the two ports are meant to be interchangeable.
//! Python's `unmask` returns an `(N, *dims)` array where this returns `N` flat volumes -
//! idiomatic per language, but the voxel ORDER is identical.

use std::path::{Path, PathBuf};

use nalgebra::DMatrix;
use num_complex::Complex64;

use crate::complex_io::{read_complex, write_complex_like, ComplexType};
use crate::pipeline::{run_complex_ica, Estimator, PipelineResult};

/// Expand masked maps back to full volume length, zero outside the mask.
///
/// `s` is `(N, V_masked)`; the returned `N` volumes each have `dims[0]*dims[1]*dims[2]`
/// entries, with column `c` of `s` scattered to the `c`-th set flat index of `mask`.
pub fn unmask(s: &DMatrix<Complex64>, mask: &[bool], dims: [usize; 3]) -> Vec<Vec<Complex64>> {
    let v = dims[0] * dims[1] * dims[2];
    assert_eq!(mask.len(), v, "mask length does not match dims");
    let kept: Vec<usize> = (0..v).filter(|&j| mask[j]).collect();
    assert_eq!(kept.len(), s.ncols(), "maps do not match the mask");

    (0..s.nrows())
        .map(|i| {
            let mut vol = vec![Complex64::new(0.0, 0.0); v];
            for (c, &j) in kept.iter().enumerate() {
                vol[j] = s[(i, c)];
            }
            vol
        })
        .collect()
}

/// Run group complex ICA over subjects given as two-file complex NIfTI pairs, and write
/// the group component maps back out as complex NIfTI pairs.
///
/// Returns the `PipelineResult` and one `(first, second)` path pair per component.
pub fn run_from_files(
    pairs: &[(PathBuf, PathBuf)],
    n_components: usize,
    out_dir: &Path,
    kind: ComplexType,
    estimator: Estimator,
    nf_dir: &Path,
) -> Result<(PipelineResult, Vec<(PathBuf, PathBuf)>), String> {
    if pairs.is_empty() {
        return Err("run_from_files: no subjects".into());
    }

    let mut subjects = Vec::with_capacity(pairs.len());
    let mut dims3: Option<[usize; 3]> = None;

    for (first, second) in pairs {
        let (data, d) = read_complex(first, second, kind)?;
        let this = [d[0], d[1], d[2]];
        match dims3 {
            None => dims3 = Some(this),
            Some(prev) if prev != this => {
                return Err(format!(
                    "subjects disagree on volume shape: {prev:?} vs {this:?}"
                ))
            }
            _ => {}
        }
        let v = this[0] * this[1] * this[2];
        let t = d[3];
        // `read_complex` returns the 4-D volume in C (standard) layout, so the LAST axis is
        // fastest: flat = ((x*ny + y)*nz + z)*nt + tt. TIME IS FASTEST, not slowest.
        // Indexing this as `data[tt * v + vv]` reads a scrambled volume with no error and
        // no shape change. Pinned by `reads_a_4d_volume_in_c_order_with_time_fastest` in
        // tests/complex_io.rs, and matches the Python port's `z.reshape(V, T).T`.
        let mut m = DMatrix::<Complex64>::zeros(t, v);
        for tt in 0..t {
            for vv in 0..v {
                m[(tt, vv)] = data[vv * t + tt];
            }
        }
        subjects.push(m);
    }
    let dims = dims3.expect("at least one subject");

    let res = run_complex_ica(&subjects, n_components, None, estimator, nf_dir)?;

    let vols = unmask(&res.s_group, &res.mask, dims);
    std::fs::create_dir_all(out_dir)
        .map_err(|e| format!("cannot create {}: {e}", out_dir.display()))?;

    let mut written = Vec::with_capacity(vols.len());
    for (k, vol) in vols.iter().enumerate() {
        let base = format!("component_{:03}.nii", k + 1);
        let first = out_dir.join(format!("R_{base}"));
        let second = out_dir.join(format!("I_{base}"));
        // Carry the first subject's geometry onto the maps, matching Python's
        // `write_complex(..., affine=affine)`. Without a reference the outputs get an
        // identity affine and will not overlay on the subject's anatomy.
        write_complex_like(vol, dims, &first, &second, kind, Some(&pairs[0].0))?;
        written.push((first, second));
    }

    Ok((res, written))
}
