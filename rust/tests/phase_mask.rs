mod common;

use common::{load_c64, load_f64, load_u8};
use complex_gift::phase_mask::{otsu_threshold, phase_quality_mask, quality_map};

#[test]
fn quality_map_matches_python_elementwise() {
    // Pure arithmetic => elementwise-comparable against the Python oracle.
    let z = load_c64("parity_mask_Z");       // (500, 80)
    let q_py = load_f64("parity_mask_Q");    // (1, 500)
    let q = quality_map(&z);
    assert_eq!(q.len(), q_py.ncols());
    for v in 0..q.len() {
        assert!(
            (q[v] - q_py[(0, v)]).abs() < 1e-12,
            "voxel {v}: rust {} vs python {}",
            q[v],
            q_py[(0, v)]
        );
    }
}

#[test]
fn mask_and_threshold_match_python() {
    let z = load_c64("parity_mask_Z");
    let tau_py = load_f64("parity_mask_tau")[(0, 0)];
    let mask_py = load_u8("parity_mask_mask");

    let (mask, _q, tau) = phase_quality_mask(&z, None);
    assert!(
        (tau - tau_py).abs() < 1e-6,
        "otsu threshold diverged: rust {tau} vs python {tau_py}"
    );
    let disagreements = mask
        .iter()
        .zip(mask_py.iter())
        .filter(|(&r, &p)| r != (p == 1))
        .count();
    assert_eq!(disagreements, 0, "mask disagrees with Python on {disagreements} voxels");
}

#[test]
fn quality_map_is_invariant_to_a_constant_per_voxel_phase() {
    // THE defining property: a static per-voxel phase offset (B0/receiver phase) must not
    // change Q at all. This is what lets the mask run before background-phase removal.
    let z = load_c64("parity_mask_Z");
    let q1 = quality_map(&z);

    let mut z2 = z.clone();
    for (v, mut row) in z2.row_iter_mut().enumerate() {
        // one constant phase per voxel, applied to that voxel's whole time series
        let phi = 0.7 * v as f64;
        let rot = num_complex::Complex64::new(phi.cos(), phi.sin());
        row.iter_mut().for_each(|x| *x *= rot);
    }
    let q2 = quality_map(&z2);
    for v in 0..q1.len() {
        assert!((q1[v] - q2[v]).abs() < 1e-12, "Q changed at voxel {v}");
    }
}

#[test]
fn otsu_discriminates_two_well_separated_modes() {
    let mut x = Vec::new();
    for i in 0..500 {
        x.push(0.10 + 0.001 * (i % 20) as f64); // lower mode, mean ~0.1095
    }
    for i in 0..500 {
        x.push(0.90 + 0.001 * (i % 20) as f64); // upper mode, mean ~0.9095
    }
    let tau = otsu_threshold(&x);

    // This is a HISTOGRAM method (256 bins) and it returns a BIN CENTRE. With a clean
    // bimodal input the between-class variance is FLAT from the lower mode's bin all the
    // way across the empty gap, and argmax takes the FIRST maximiser - so tau is the centre
    // of a bin that can sit INSIDE the lower mode. That is not a bug: skimage's
    // threshold_otsu returns the identical value on this input (verified bit-for-bit), and
    // the Python/Rust parity test above pins them together to 1.7e-16.
    //
    // So do not assert "the whole lower mode is below tau" - a binned Otsu cannot promise
    // that. Assert what it does promise: the threshold discriminates the two modes.
    let mean_lo = x[..500].iter().sum::<f64>() / 500.0;
    let mean_hi = x[500..].iter().sum::<f64>() / 500.0;
    assert!(
        mean_lo < tau && tau < mean_hi,
        "tau={tau} must fall between the modes ({mean_lo} .. {mean_hi})"
    );
    assert!(
        x[500..].iter().all(|&v| v > tau),
        "the entire upper mode must lie above tau"
    );
}
