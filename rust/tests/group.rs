mod common;

use common::{match_sources, TestRng};
use complex_gift::group::{back_reconstruct, two_stage_pca};
use complex_gift::whiten::whiten_hermitian;
use nalgebra::DMatrix;
use num_complex::Complex64;

/// Build `n_sub` subjects that genuinely DIFFER: shared maps plus a per-subject
/// perturbation, plus subject-specific noise (which also lifts the rank above N so a
/// reduction can legitimately keep n_subject > n_components).
fn make_subjects(
    rng: &mut TestRng,
    n: usize,
    v: usize,
    t: usize,
    n_sub: usize,
) -> (Vec<DMatrix<Complex64>>, Vec<DMatrix<Complex64>>) {
    let smaps = DMatrix::<f64>::from_fn(n, v, |_, _| {
        rng.normal() * rng.normal().abs().powf(1.5)
    });
    let phi: Vec<f64> = (0..v).map(|_| (rng.uniform() * 20.0 - 10.0) / 180.0 * std::f64::consts::PI).collect();

    let mut subjects = Vec::new();
    let mut truths = Vec::new();
    for _ in 0..n_sub {
        let si = DMatrix::<f64>::from_fn(n, v, |i, j| {
            smaps[(i, j)] + 0.5 * rng.normal() * rng.normal().abs().powf(1.5)
        });
        let tc = DMatrix::<f64>::from_fn(t, n, |_, _| rng.normal());

        // z(v,t) = rho(v,t) * e^{i phi(v)}: POSITIVE magnitude (static baseline + BOLD-like
        // modulation + subject noise) times a STATIC per-voxel background phase.
        let modulation = &tc * &si; // (t, v)
        let mut z = DMatrix::<Complex64>::zeros(t, v);
        for tt in 0..t {
            for vv in 0..v {
                let mag = 100.0 + 5.0 * modulation[(tt, vv)] + 2.0 * rng.normal();
                let rot = Complex64::new(phi[vv].cos(), phi[vv].sin());
                z[(tt, vv)] = Complex64::new(mag, 0.0) * rot;
            }
        }
        subjects.push(z);

        let mut strue = DMatrix::<Complex64>::zeros(n, v);
        for i in 0..n {
            for j in 0..v {
                let rot = Complex64::new(phi[j].cos(), phi[j].sin());
                strue[(i, j)] = Complex64::new(si[(i, j)], 0.0) * rot;
            }
        }
        truths.push(strue);
    }
    (subjects, truths)
}

#[test]
fn two_stage_pca_reduces_to_the_requested_order() {
    let mut rng = TestRng::new(3);
    let (subjects, _) = make_subjects(&mut rng, 3, 400, 60, 4);
    let red = two_stage_pca(&subjects, 6, 3).expect("reduction");
    assert_eq!(red.xg.nrows(), 3);
    assert_eq!(red.xg.ncols(), 400);
    assert_eq!(red.reduced.len(), 4);
    assert_eq!(red.reduced[0].shape(), (6, 400));
}

#[test]
fn back_reconstruction_is_subject_specific() {
    // Guard against the silent no-op: if back_reconstruct collapsed to S_group, every
    // subject would get identical maps and this assertion would fail.
    let mut rng = TestRng::new(4);
    let (n, v, t, n_sub) = (3usize, 400usize, 60usize, 4usize);
    let (subjects, truths) = make_subjects(&mut rng, n, v, t, n_sub);

    let red = two_stage_pca(&subjects, 6, n).expect("reduction");

    // stand in for ICA with a plain whitening of the group data: back-reconstruction is a
    // linear-algebra property and does not depend on WHICH unmixing we use.
    let w = whiten_hermitian(&red.xg, n).expect("whiten");
    let s_group = w.xw.clone();
    let a_group = w.w_dewhiten.clone();

    let subs = back_reconstruct(&a_group, &red);
    assert_eq!(subs.len(), n_sub);

    for (i, (s_i, a_i)) in subs.iter().enumerate() {
        assert_eq!(s_i.shape(), (n, v));
        assert_eq!(a_i.nrows(), t);

        let (_, c_subject) = match_sources(s_i, &truths[i]);
        let (_, c_group) = match_sources(&s_group, &truths[i]);
        let mean_subject: f64 = c_subject.iter().sum::<f64>() / n as f64;
        let mean_group: f64 = c_group.iter().sum::<f64>() / n as f64;
        assert!(
            mean_subject > mean_group,
            "subject {i}: back-reconstruction ({mean_subject:.4}) must explain the subject \
             better than the group maps ({mean_group:.4}) - a collapse to S_group would \
             make these equal"
        );
    }
}
