mod common;

use common::{isi, load_c64, match_sources};
use complex_gift::estimators::nc_fastica::nc_fastica;

#[test]
fn separates_the_shared_fixture_as_well_as_the_matlab_reference() {
    // nc-FastICA is deterministic, but its internal eigendecomposition means the component
    // ORDER and PHASE may differ from Python/MATLAB. Compare with ISI (permutation- and
    // phase-invariant) against the KNOWN true mixing - never elementwise.
    let cx = load_c64("cX");
    let a_true = load_c64("A");
    let cs = load_c64("cS");
    let w_matlab = load_c64("oracle_ncfastica_W");

    let isi_matlab = isi(&(&w_matlab * &a_true));

    let res = nc_fastica(&cx, "log", 1e-5, None).expect("nc_fastica");
    assert_eq!(res.w.shape(), (6, 6));
    assert_eq!(res.s.shape(), (6, cx.ncols()));

    let isi_rust = isi(&(&res.w * &a_true));
    assert!(isi_rust < 0.05, "port separates poorly: ISI={isi_rust:.5}");

    // nc-FastICA is FULLY DETERMINISTIC (the reference has no rand/randn), so this is not a
    // "close enough" comparison - the port reproduces MATLAB's ISI to every printed digit
    // and the bound says so. The old bound (2*isi_matlab + 0.01) allowed 3.4x drift, which
    // is dishonest for a deterministic estimator AND left real bugs invisible: corrupting
    // the pseudo-covariance to the conjugate transpose merely doubles ISI to ~0.0131, which
    // the loose bound absorbed silently. A near-equality bound is what gives that mutation
    // teeth here.
    assert!(
        (isi_rust - isi_matlab).abs() < 1e-6,
        "deterministic port must reproduce MATLAB's ISI: rust={isi_rust:.12} matlab={isi_matlab:.12}"
    );

    let (perm, corr) = match_sources(&res.s, &cs);
    let mut seen = perm.clone();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), 6, "not a bijective permutation: {perm:?}");
    assert!(corr.iter().all(|&c| c > 0.9), "weak recovery: {corr:?}");
}

#[test]
fn default_max_iter_is_the_reference_cap_of_15n() {
    use complex_gift::estimators::nc_fastica::resolve_max_iter;
    // MATLAB's maxcounter=50 is dead code; the real cap is 15*n. The shared fixture
    // converges in ~16 iterations, so neither cap ever bites and comparing OUTPUTS
    // cannot tell 50 from 90 - assert the resolution itself, which a regression to
    // unwrap_or(50) would fail immediately.
    assert_eq!(resolve_max_iter(None, 6), 90);
    assert_eq!(resolve_max_iter(None, 4), 60);
    assert_eq!(resolve_max_iter(Some(7), 6), 7);
}

#[test]
fn rejects_an_unknown_nonlinearity() {
    let cx = load_c64("cX");
    assert!(nc_fastica(&cx, "bogus", 1e-5, None).is_err());
}
