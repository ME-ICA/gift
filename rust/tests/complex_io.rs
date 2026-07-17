mod common;

use std::path::Path;

use gift_rs::complex_io::{complex_file_pair, read_complex, ComplexType};

// The fixtures directory holds two small NIfTI volumes written by nibabel for this test:
// complex_ica_fixtures/npy/../nifti/{R_probe.nii, I_probe.nii} (see Step 3).
fn nifti_dir() -> std::path::PathBuf {
    common::fixtures_dir().parent().unwrap().join("nifti")
}

#[test]
fn reads_a_real_imaginary_pair() {
    let d = nifti_dir();
    let (data, dims) = read_complex(
        &d.join("R_probe.nii"),
        &d.join("I_probe.nii"),
        ComplexType::RealImag,
    )
    .expect("read");
    assert_eq!(dims[0] * dims[1] * dims[2] * dims[3], data.len());
    // the probe volume was written as re = index, im = -index
    assert!((data[5].re - 5.0).abs() < 1e-9);
    assert!((data[5].im + 5.0).abs() < 1e-9);
}

#[test]
fn file_pair_naming_follows_the_gift_convention() {
    let (a, b) = complex_file_pair(Path::new("/data/sub01_run1.nii"), ("R_", "I_")).unwrap();
    assert_eq!(a.file_name().unwrap(), "R_sub01_run1.nii");
    assert_eq!(b.file_name().unwrap(), "I_sub01_run1.nii");
}

#[test]
fn file_pair_requires_an_underscore() {
    let err = complex_file_pair(Path::new("/data/sub01.nii"), ("R_", "I_")).unwrap_err();
    assert!(err.contains("underscore"), "unhelpful error: {err}");
}

#[test]
fn write_then_read_roundtrips() {
    use gift_rs::complex_io::write_complex;
    use num_complex::Complex64;

    let dir = std::env::temp_dir().join("cg_rust_io_roundtrip");
    std::fs::create_dir_all(&dir).unwrap();
    let dims = [4usize, 4, 2];
    let n = dims[0] * dims[1] * dims[2];
    let data: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new(i as f64 * 0.5, -(i as f64) * 0.25))
        .collect();

    let (f1, f2) = (dir.join("R_out.nii"), dir.join("I_out.nii"));
    write_complex(&data, dims, &f1, &f2, ComplexType::RealImag).expect("write");

    let (back, d) = read_complex(&f1, &f2, ComplexType::RealImag).expect("read");
    assert_eq!([d[0], d[1], d[2]], dims);
    for i in 0..n {
        assert!(
            (back[i] - data[i]).norm() < 1e-9,
            "roundtrip lost value at {i}"
        );
    }
}

/// Task 11's driver is the first consumer to flatten a 4-D volume, and the ordering it
/// assumes is invisible to the type system: reading a (nx,ny,nz,nt) volume as time-slowest
/// instead of time-fastest yields a same-shaped, silently scrambled matrix.
///
/// So pin the contract here, at its source: `read_complex` returns the volume in C
/// (standard) layout over ALL FOUR axes, i.e. flat = ((x*ny + y)*nz + z)*nt + t. Time is the
/// FASTEST axis; the flat VOXEL index is vv = x*(ny*nz) + y*nz + z, with x SLOWEST.
/// (`read_volume` calls `.as_standard_layout()` for exactly this reason.)
#[test]
fn reads_a_4d_volume_in_c_order_with_time_fastest() {
    use nifti::writer::WriterOptions;

    let (nx, ny, nz, nt) = (3usize, 2, 4, 5);
    let dir = std::env::temp_dir().join("cg_rust_io_4d");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // Every element gets a UNIQUE value encoding its own 4-D C-order position, so any
    // transposition, axis swap or partial scramble changes at least one value.
    let mut re = ndarray::Array4::<f64>::zeros((nx, ny, nz, nt));
    let mut im = ndarray::Array4::<f64>::zeros((nx, ny, nz, nt));
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                for t in 0..nt {
                    re[(x, y, z, t)] = (((x * ny + y) * nz + z) * nt + t) as f64;
                    im[(x, y, z, t)] = (x * 100 + y * 10 + z) as f64; // voxel identity only
                }
            }
        }
    }
    let f1 = dir.join("R_probe4d.nii");
    let f2 = dir.join("I_probe4d.nii");
    WriterOptions::new(&f1).write_nifti(&re).unwrap();
    WriterOptions::new(&f2).write_nifti(&im).unwrap();

    let (data, d) = read_complex(&f1, &f2, ComplexType::RealImag).expect("read 4-D");
    assert_eq!(d, [nx, ny, nz, nt]);
    assert_eq!(data.len(), nx * ny * nz * nt);

    // the flat index IS the C-order 4-D index
    for (i, z) in data.iter().enumerate() {
        assert!(
            (z.re - i as f64).abs() < 1e-9,
            "flat position {i} holds {} - read_complex is not returning C order",
            z.re
        );
    }

    // and therefore data[vv * nt + tt] is voxel vv at time tt, with vv C-order over (x,y,z)
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                let vv = x * (ny * nz) + y * nz + z;
                for tt in 0..nt {
                    assert!(
                        (data[vv * nt + tt].im - (x * 100 + y * 10 + z) as f64).abs() < 1e-9,
                        "voxel ({x},{y},{z}) is not at flat voxel index {vv}"
                    );
                }
            }
        }
    }
}
