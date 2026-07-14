mod common;

use common::{isi, load_c64, match_sources};
use nalgebra::DMatrix;
use num_complex::Complex64;

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

#[test]
fn isi_is_zero_for_a_scaled_permutation() {
    // identity
    let eye = DMatrix::<Complex64>::identity(4, 4);
    assert!(isi(&eye) < 1e-12);

    // a genuinely permuted, arbitrarily phase-scaled global matrix is ALSO perfect
    // separation - that is exactly ICA's indeterminacy, and ISI must be blind to it.
    let perm = [2usize, 0, 3, 1];
    let mut g = DMatrix::<Complex64>::zeros(4, 4);
    for (i, &j) in perm.iter().enumerate() {
        let k = i as f64;
        g[(i, j)] = c((k + 1.0) * (0.3 * k).cos(), (k + 1.0) * (0.3 * k).sin());
    }
    assert!(isi(&g) < 1e-12, "isi = {}", isi(&g));
}

#[test]
fn isi_is_large_for_a_maximally_mixed_matrix() {
    let g = DMatrix::from_element(4, 4, c(1.0, 0.0));
    assert!(isi(&g) > 0.9, "isi = {}", isi(&g));
}

#[test]
fn match_sources_recovers_permutation_through_phase_and_scale() {
    let s = load_c64("cS"); // (6, 4000) ground truth
    let n = s.nrows();
    let perm_true = [2usize, 0, 4, 1, 5, 3];
    let mut est = DMatrix::<Complex64>::zeros(n, s.ncols());
    for (k, &j) in perm_true.iter().enumerate() {
        let scale = c(2.0 * (0.7 * k as f64).cos(), 2.0 * (0.7 * k as f64).sin());
        for t in 0..s.ncols() {
            est[(k, t)] = s[(j, t)] * scale;
        }
    }
    let (perm, corr) = match_sources(&est, &s);
    assert_eq!(perm, perm_true.to_vec());
    assert!(corr.iter().all(|&x| x > 0.99), "corr = {:?}", corr);
}

#[test]
fn fixtures_load_and_satisfy_the_ground_truth_identity() {
    let cs = load_c64("cS");
    let a = load_c64("A");
    let cx = load_c64("cX");
    assert_eq!(cs.shape(), (6, 4000));
    assert_eq!(a.shape(), (6, 6));
    // the whole oracle rests on cX == A @ cS
    let recon = &a * &cs;
    let err = (&recon - &cx).norm() / cx.norm();
    assert!(err < 1e-12, "cX != A @ cS (rel err {err:.3e})");
}
