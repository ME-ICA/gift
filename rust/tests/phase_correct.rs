mod common;

use common::{load_c64, load_f64, load_u8};
use complex_gift::phase_correct::{align_to_reference, correct_phase};

#[test]
fn correct_phase_matches_python_elementwise() {
    // Pure arithmetic given fixed inputs => elementwise-comparable against Python.
    let mut s = load_c64("parity_pc_S_in");
    let mut a = load_c64("parity_pc_A_in");
    let mask: Vec<bool> = load_u8("parity_pc_mask").iter().map(|&b| b == 1).collect();

    let s_py = load_c64("parity_pc_S_out");
    let a_py = load_c64("parity_pc_A_out");
    let theta_py = load_f64("parity_pc_theta");

    let theta = correct_phase(&mut s, &mut a, Some(&mask));

    for k in 0..theta.len() {
        assert!(
            (theta[k] - theta_py[(0, k)]).abs() < 1e-10,
            "theta[{k}]: rust {} vs python {}",
            theta[k],
            theta_py[(0, k)]
        );
    }
    assert!((&s - &s_py).norm() / s_py.norm() < 1e-10, "S diverged from Python");
    assert!((&a - &a_py).norm() / a_py.norm() < 1e-10, "A diverged from Python");
}

#[test]
fn correct_phase_preserves_the_reconstruction() {
    // The whole point: rotating S by e^{i0} and A by e^{-i0} must leave A*S untouched -
    // including when the residual pi sign-flip branch fires.
    let mut s = load_c64("parity_pc_S_in");
    let mut a = load_c64("parity_pc_A_in");
    let before = &a * &s;

    let _ = correct_phase(&mut s, &mut a, None);

    let after = &a * &s;
    let err = (&before - &after).norm() / before.norm();
    assert!(err < 1e-10, "A*S was not preserved (rel err {err:.3e})");
}

#[test]
fn align_to_reference_matches_python_and_collapses_known_rotations() {
    let s_ref = load_c64("parity_align_S_ref");
    let mut s = load_c64("parity_align_S_in");
    let s_py = load_c64("parity_align_S_out");
    let theta_py = load_f64("parity_align_theta");

    let theta = align_to_reference(&mut s, &s_ref);

    for k in 0..theta.len() {
        assert!(
            (theta[k] - theta_py[(0, k)]).abs() < 1e-10,
            "theta[{k}]: rust {} vs python {}",
            theta[k],
            theta_py[(0, k)]
        );
    }
    assert!((&s - &s_py).norm() / s_py.norm() < 1e-10, "aligned S diverged from Python");
}
