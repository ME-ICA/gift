mod common;

use common::{fixtures_dir, load_f64};
use complex_gift::nf_table::{load_nf_table, simplified_ppval};

#[test]
fn tables_have_the_expected_structure() {
    let nf = load_nf_table(&fixtures_dir());
    for (i, v) in nf.iter().enumerate() {
        assert_eq!(v.pp.breaks.len(), v.pp.pieces + 1, "nf{}", i + 1);
        assert_eq!(v.pp.coefs.len(), v.pp.pieces, "nf{}", i + 1);
        assert!(v.max_egx > v.min_egx, "nf{}", i + 1);
    }
    // only nf1 carries critical_point2; nf2-nf8 are NaN by design (inert legacy metadata)
    assert!(nf[0].critical_point2.is_finite());
    assert!(nf[1].critical_point2.is_nan());
}

#[test]
fn ppval_matches_python_elementwise() {
    // Pure arithmetic => this IS elementwise-comparable against the Python oracle.
    // The sampled xs deliberately run outside the knot range, exercising the clamp
    // branches of the binary search as well as the interior.
    let nf = load_nf_table(&fixtures_dir());
    let xs = load_f64("parity_ppval_xs"); // (8, 101)
    let ys = load_f64("parity_ppval_ys"); // (8, 101)

    for k in 0..8 {
        for j in 0..xs.ncols() {
            let got = simplified_ppval(&nf[k].pp, xs[(k, j)]);
            let want = ys[(k, j)];
            let tol = 1e-10 * want.abs().max(1.0);
            assert!(
                (got - want).abs() < tol,
                "nf{} at x={}: rust {} vs python {}",
                k + 1,
                xs[(k, j)],
                got,
                want
            );
        }
    }
}
