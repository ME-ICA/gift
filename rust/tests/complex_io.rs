mod common;

use std::path::Path;

use complex_gift::complex_io::{complex_file_pair, read_complex, ComplexType};

// The fixtures directory holds two small NIfTI volumes written by nibabel for this test:
// complex_ica_fixtures/npy/../nifti/{R_probe.nii, I_probe.nii} (see Step 3).
fn nifti_dir() -> std::path::PathBuf {
    common::fixtures_dir().parent().unwrap().join("nifti")
}

#[test]
fn reads_a_real_imaginary_pair() {
    let d = nifti_dir();
    let (data, dims) =
        read_complex(&d.join("R_probe.nii"), &d.join("I_probe.nii"), ComplexType::RealImag)
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
    use complex_gift::complex_io::write_complex;
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
        assert!((back[i] - data[i]).norm() < 1e-9, "roundtrip lost value at {i}");
    }
}

