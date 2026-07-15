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
fn default_max_iter_is_the_reference_cap_of_15n() {
    // MATLAB's `maxcounter = 50` is DEAD CODE; the real cap is 15*n. Guard the default so
    // nobody "restores" 50 and silently stops up to 3x early for realistic model orders.
    let cx = load_c64("cX"); // n = 6 -> 15*n = 90 != 50
    let a = nc_fastica(&cx, "log", 1e-5, None).expect("default");
    let b = nc_fastica(&cx, "log", 1e-5, Some(15 * cx.nrows())).expect("explicit 15n");
    assert!((&a.w - &b.w).norm() < 1e-12, "default cap is not 15*n");
}

#[test]
fn rejects_an_unknown_nonlinearity() {
    let cx = load_c64("cX");
    assert!(nc_fastica(&cx, "bogus", 1e-5, None).is_err());
}
