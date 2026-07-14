mod common;

use common::load_c64;
use complex_gift::whiten::whiten_hermitian;
use nalgebra::DMatrix;
use num_complex::Complex64;

#[test]
fn whitening_makes_the_covariance_identity() {
    let x = load_c64("cX"); // (6, 4000)
    let n = 6;
    let w = whiten_hermitian(&x, n).expect("whiten");
    assert_eq!(w.xw.shape(), (n, x.ncols()));

    let t = x.ncols() as f64;
    let cov = (&w.xw * w.xw.adjoint()) / Complex64::new(t, 0.0);
    let err = (&cov - DMatrix::<Complex64>::identity(n, n)).norm();
    assert!(err < 1e-8, "whitened covariance is not identity (err {err:.3e})");
}

#[test]
fn dewhitening_reconstructs_the_centred_data() {
    let x = load_c64("cX");
    let n = 6; // data is exactly rank 6, so dewhitening must reconstruct it
    let w = whiten_hermitian(&x, n).expect("whiten");

    // centre x along its own axis-1 (the sample axis), as whiten_hermitian does
    let t = x.ncols();
    let mut xc = x.clone();
    for mut row in xc.row_iter_mut() {
        let mean: Complex64 = row.iter().sum::<Complex64>() / Complex64::new(t as f64, 0.0);
        row.iter_mut().for_each(|z| *z -= mean);
    }

    let recon = &w.w_dewhiten * &w.xw;
    let err = (&recon - &xc).norm() / xc.norm();
    assert!(err < 1e-10, "dewhitening did not reconstruct (rel err {err:.3e})");
}

#[test]
fn errors_when_more_components_than_rows() {
    let x = load_c64("cX"); // 6 rows
    assert!(whiten_hermitian(&x, 12).is_err());
}

#[test]
fn errors_on_rank_deficient_input() {
    // 6 rows spanning only 3 independent directions -> asking for 6 must fail loudly
    // rather than returning NaNs from sqrt of a ~0 eigenvalue.
    let base = load_c64("cS").rows(0, 3).into_owned(); // (3, 4000)
    let mut x = DMatrix::<Complex64>::zeros(6, base.ncols());
    for i in 0..6 {
        let src = i % 3;
        let scale = Complex64::new(1.0 + i as f64, 0.5 * i as f64);
        for j in 0..base.ncols() {
            x[(i, j)] = base[(src, j)] * scale;
        }
    }
    assert!(whiten_hermitian(&x, 6).is_err());
}
