mod common;

use common::{fixtures_dir, isi, load_c64, match_sources};
use gift_rs::estimators::cebm::{cebm, pseudo_cov};
use nalgebra::DMatrix;
use num_complex::Complex64;

#[test]
fn is_reproducible_for_a_fixed_seed() {
    let cx = load_c64("cX");
    let a = cebm(&cx, &fixtures_dir(), 0, 1e-4, None).expect("cebm");
    let b = cebm(&cx, &fixtures_dir(), 0, 1e-4, None).expect("cebm");
    assert!(
        (&a.w - &b.w).norm() < 1e-12,
        "same seed must give the same W"
    );
}

#[test]
fn pseudo_cov_uses_the_plain_transpose() {
    // Guard the single most likely silent bug. A Phase-2 review PROVED that swapping this
    // to the conjugate transpose still passes the ISI bar - so ISI alone cannot catch it.
    let x = load_c64("cS").rows(0, 3).into_owned();
    let t = x.ncols() as f64;

    let got = pseudo_cov(&x);
    let want_plain = (&x * x.transpose()) / Complex64::new(t, 0.0);
    let want_conj = (&x * x.adjoint()) / Complex64::new(t, 0.0);

    assert!(
        (&got - &want_plain).norm() < 1e-9,
        "pseudo-cov must use the PLAIN transpose"
    );
    assert!(
        (&got - &want_conj).norm() > 1e-3,
        "pseudo-cov must NOT be the Hermitian covariance"
    );
}

#[test]
fn separates_the_shared_fixture_as_well_as_the_matlab_reference() {
    // CEBM is stochastic, so MATLAB's W is NOT reproducible elementwise. The meaningful
    // question is whether the port separates the fixture as well as the reference does,
    // measured against the KNOWN true mixing A.
    let cx = load_c64("cX");
    let a_true = load_c64("A");
    let cs = load_c64("cS");
    let w_matlab = load_c64("oracle_cebm_W");

    let isi_matlab = isi(&(&w_matlab * &a_true));

    let res = cebm(&cx, &fixtures_dir(), 0, 1e-4, None).expect("cebm");
    let isi_rust = isi(&(&res.w * &a_true));

    assert!(isi_rust < 0.05, "port separates poorly: ISI={isi_rust:.5}");
    assert!(
        isi_rust < 2.0 * isi_matlab + 0.01,
        "port is materially worse than MATLAB: rust={isi_rust:.5} matlab={isi_matlab:.5}"
    );

    let (perm, corr) = match_sources(&res.s, &cs);
    let mut seen = perm.clone();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), 6, "not a bijective permutation: {perm:?}");
    assert!(corr.iter().all(|&c| c > 0.9), "weak recovery: {corr:?}");
}

#[test]
fn separates_at_n_10_exercising_the_incremental_branch() {
    // N > 7 takes the incremental Sherman-Morrison inv_Q path - the one real fMRI runs
    // use, and the one the N=6 fixture never reaches.
    let n = 10usize;
    let t = 5000usize;
    let mut rng = common::TestRng::new(7);

    // super-Gaussian sources: ICA cannot separate Gaussian sources at all
    let mut s = DMatrix::<Complex64>::zeros(n, t);
    for i in 0..n {
        for j in 0..t {
            let re = rng.normal() * rng.normal().abs().powf(1.5);
            let im = rng.normal() * rng.normal().abs().powf(1.5);
            s[(i, j)] = Complex64::new(re, im);
        }
    }
    let a_mix =
        DMatrix::<Complex64>::from_fn(n, n, |_, _| Complex64::new(rng.normal(), rng.normal()));
    let x = &a_mix * &s;

    let res = cebm(&x, &fixtures_dir(), 0, 1e-4, None).expect("cebm at n=10");
    let isi_rust = isi(&(&res.w * &a_mix));
    assert!(
        isi_rust < 0.05,
        "n=10 (incremental branch) ISI={isi_rust:.5}"
    );
}
