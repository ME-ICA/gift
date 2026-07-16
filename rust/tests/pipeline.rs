mod common;

use common::{fixtures_dir, match_sources, TestRng};
use complex_gift::pipeline::{align_subject, run_complex_ica, Estimator};
use nalgebra::DMatrix;
use num_complex::Complex64;

#[test]
fn end_to_end_group_complex_ica() {
    let mut rng = TestRng::new(21);
    let (n, v_sig, v_noise, t, n_sub) = (3usize, 1000usize, 300usize, 60usize, 4usize);
    let v = v_sig + v_noise;

    // super-Gaussian spatial maps (ICA cannot separate Gaussian sources at all)
    let smaps =
        DMatrix::<f64>::from_fn(n, v_sig, |_, _| rng.normal() * rng.normal().abs().powf(1.5));
    let scale = smaps.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
    let smaps = smaps / scale;

    // static per-voxel background phase, small spread (+/-10 deg)
    let phi: Vec<f64> = (0..v_sig)
        .map(|_| (rng.uniform() * 20.0 - 10.0) / 180.0 * std::f64::consts::PI)
        .collect();

    let mut subjects = Vec::new();
    for _ in 0..n_sub {
        let tc = DMatrix::<f64>::from_fn(t, n, |_, _| rng.normal());
        let modulation = &tc * &smaps; // (t, v_sig)
        let mut z = DMatrix::<Complex64>::zeros(t, v);
        for tt in 0..t {
            // signal voxels: POSITIVE magnitude x STATIC per-voxel phase -> phase-stable
            for vv in 0..v_sig {
                let mag = 100.0 + 10.0 * rng.uniform() + 5.0 * modulation[(tt, vv)];
                assert!(mag > 0.0);
                z[(tt, vv)] =
                    Complex64::new(mag, 0.0) * Complex64::new(phi[vv].cos(), phi[vv].sin());
            }
            // noise voxels: random phase per timepoint -> the mask must drop them
            for vv in v_sig..v {
                let mag = 0.5 + rng.uniform();
                let ang = std::f64::consts::TAU * rng.uniform();
                z[(tt, vv)] = Complex64::new(mag * ang.cos(), mag * ang.sin());
            }
        }
        subjects.push(z);
    }

    let res = run_complex_ica(
        &subjects,
        n,
        None,
        Estimator::Cebm { seed: 0 },
        &fixtures_dir(),
    )
    .expect("pipeline");

    // the phase mask kept the signal voxels and dropped the random-phase ones
    let kept_sig = res.mask[..v_sig].iter().filter(|&&m| m).count() as f64 / v_sig as f64;
    let kept_noise = res.mask[v_sig..].iter().filter(|&&m| m).count() as f64 / v_noise as f64;
    assert!(
        kept_sig > 0.9,
        "mask dropped signal voxels: kept {kept_sig:.3}"
    );
    assert!(
        kept_noise < 0.05,
        "mask kept noise voxels: kept {kept_noise:.3}"
    );

    // the group decomposition recovered the true maps (up to permutation/phase).
    // Build the truth over ALL voxels (noise voxels carry no signal) and apply the SAME
    // mask, so a leaked noise voxel cannot cause a shape mismatch.
    let mut strue_full = DMatrix::<Complex64>::zeros(n, v);
    for i in 0..n {
        for j in 0..v_sig {
            strue_full[(i, j)] =
                Complex64::new(smaps[(i, j)], 0.0) * Complex64::new(phi[j].cos(), phi[j].sin());
        }
    }
    let kept: Vec<usize> = (0..v).filter(|&j| res.mask[j]).collect();
    let mut strue = DMatrix::<Complex64>::zeros(n, kept.len());
    for (c, &j) in kept.iter().enumerate() {
        for i in 0..n {
            strue[(i, c)] = strue_full[(i, j)];
        }
    }

    let (perm, corr) = match_sources(&res.s_group, &strue);
    let mut seen = perm.clone();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), n, "not a bijective permutation: {perm:?}");
    assert!(corr.iter().all(|&c| c > 0.8), "weak recovery: {corr:?}");

    // phase correction left the maps concentrated on the real axis
    for i in 0..n {
        let (mut im, mut tot) = (0.0f64, 0.0f64);
        for j in 0..res.s_group.ncols() {
            im += res.s_group[(i, j)].im.powi(2);
            tot += res.s_group[(i, j)].norm_sqr();
        }
        assert!(im / tot < 0.15, "component {i} is not real-axis aligned");
    }

    assert_eq!(res.subjects.len(), n_sub);

    // ---- step 6: back-reconstruction actually happened and is self-consistent ----
    //
    // NOTE: `assert_eq!(res.subjects.len(), n_sub)` alone is vacuous - it passes even when
    // every (S_i, A_i) is all zeros. The assertions below are what give step 6 teeth.
    let nrm = |m: &DMatrix<Complex64>| m.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();

    for (i, (s_i, a_i)) in res.subjects.iter().enumerate() {
        assert_eq!(s_i.shape(), (n, kept.len()), "subject {i} S_i shape");
        assert_eq!(a_i.shape(), (t, n), "subject {i} A_i shape");

        // Rebuild subject i's masked, per-voxel temporally de-meaned data. With
        // n_subject == n_components the subject block Bi is square and invertible, so
        // A_i * S_i is the orthogonal projection onto the rank-n signal subspace.
        //
        // Strictly it projects the ROW-CENTERED x: whiten_hermitian (whiten.rs:32-33)
        // additionally removes each timepoint's mean over voxels. The Pythagoras identity
        // in (c) still holds against the x rebuilt here because row-centering forces
        // xc * 1 = 0, so the rank-1 cross-term the row mean would contribute vanishes
        // exactly. Do not read (c) as "p projects precisely this x" - it does not.
        let mut x = DMatrix::<Complex64>::zeros(t, kept.len());
        for (c, &j) in kept.iter().enumerate() {
            for tt in 0..t {
                x[(tt, c)] = subjects[i][(tt, j)];
            }
        }
        for mut col in x.column_iter_mut() {
            let m: Complex64 = col.iter().sum::<Complex64>() / Complex64::new(t as f64, 0.0);
            col.iter_mut().for_each(|z| *z -= m);
        }

        let prod = a_i * s_i;
        let xn = nrm(&x);
        let pn = nrm(&prod);
        let resid = nrm(&(&x - &prod));

        // (a) the product carries real energy - fails immediately on zeros/garbage
        let frac = pn / xn;
        assert!(
            frac > 0.3,
            "subject {i}: A_i*S_i carries no energy (||p||/||x|| = {frac:.4})"
        );

        // (b) <x, A_i*S_i> is real and positive. A projection cannot rotate what it keeps,
        //     so any spurious per-component phase on A_i shows up here as a nonzero arg.
        let dot: Complex64 = x.iter().zip(prod.iter()).map(|(a, b)| a.conj() * b).sum();
        assert!(
            dot.arg().abs() < 1e-6,
            "subject {i}: <x, A_i*S_i> is not real (arg = {:.3e}) - A_i and S_i disagree",
            dot.arg()
        );

        // (c) Pythagoras: ||x - p||^2 == ||x||^2 - ||p||^2 iff p really is the orthogonal
        //     projection of x. This pins A_i*S_i to the subject's ACTUAL data.
        let lhs = resid * resid;
        let rhs = xn * xn - pn * pn;
        assert!(
            (lhs - rhs).abs() / (xn * xn) < 1e-9,
            "subject {i}: A_i*S_i is not the orthogonal projection of X_i \
             (||x-p||^2 = {lhs:.6e}, ||x||^2-||p||^2 = {rhs:.6e})"
        );

        // (d) each component of S_i is phase-aligned to the group reference
        for k in 0..n {
            let d: Complex64 = (0..s_i.ncols())
                .map(|j| res.s_group[(k, j)].conj() * s_i[(k, j)])
                .sum();
            assert!(
                d.arg().abs() < 1e-6,
                "subject {i} component {k}: not aligned to the group (arg = {:.3e})",
                d.arg()
            );
        }

        // (e) S_i must carry SUBJECT-SPECIFIC variance. `pinv(Bi) * (Bi * S_group)` collapses
        //     to S_group for every subject - a silent no-op this catches.
        let dev = nrm(&(s_i - &res.s_group)) / nrm(&res.s_group);
        assert!(
            dev > 0.1,
            "subject {i}: S_i collapsed onto S_group (rel. deviation {dev:.4}) - \
             back_reconstruct is not using the subject's own reduced data"
        );
    }

    // and the subjects must differ from EACH OTHER, not just from the group
    for i in 0..n_sub {
        for j in (i + 1)..n_sub {
            let d = nrm(&(&res.subjects[i].0 - &res.subjects[j].0)) / nrm(&res.s_group);
            assert!(
                d > 0.1,
                "subjects {i} and {j} have identical maps (rel. diff {d:.4})"
            );
        }
    }
}

/// The end-to-end fixture CANNOT pin the counter-rotation sign: when the group model holds
/// and each subject block is full column rank, back-reconstructed maps already arrive in
/// the group frame, so `align_to_reference` returns theta == 0 and both signs are the same
/// code path. (Verified: flipping the sign in `align_subject` leaves the e2e output
/// byte-identical, and deleting the whole alignment step leaves the e2e test green.)
///
/// So drive it directly with a synthetic NONZERO rotation, which is the only way to make
/// the sign observable.
#[test]
fn align_subject_undoes_a_known_rotation_and_preserves_the_product() {
    let mut rng = TestRng::new(7);
    let (n, v, t) = (3usize, 40usize, 12usize);

    // an arbitrary reference and an arbitrary mixing matrix
    let s_ref =
        DMatrix::<Complex64>::from_fn(n, v, |_, _| Complex64::new(rng.normal(), rng.normal()));
    let a0 = DMatrix::<Complex64>::from_fn(t, n, |_, _| Complex64::new(rng.normal(), rng.normal()));

    // rotate each source by a KNOWN, DISTINCT, NONZERO phase
    let applied = [0.7f64, -1.9, 2.6];
    let mut s = s_ref.clone();
    for k in 0..n {
        let rot = Complex64::new(applied[k].cos(), applied[k].sin());
        for i in 0..v {
            s[(k, i)] *= rot;
        }
    }
    let mut a = a0.clone();
    // counter-rotate A so the pair starts out consistent: a * s == a0 * s_ref
    for k in 0..n {
        let inv = Complex64::new(applied[k].cos(), -applied[k].sin());
        for r in 0..t {
            a[(r, k)] *= inv;
        }
    }
    let product_before = &a * &s;

    let theta = align_subject(&mut s, &mut a, &s_ref);

    // 1. the rotation was undone: theta == -applied, and s is back on s_ref
    for k in 0..n {
        let resid = (Complex64::new(theta[k].cos(), theta[k].sin())
            * Complex64::new(applied[k].cos(), applied[k].sin())
            - Complex64::new(1.0, 0.0))
        .norm();
        assert!(
            resid < 1e-9,
            "component {k}: theta {} does not undo the applied rotation {}",
            theta[k],
            applied[k]
        );
    }
    let s_err = (&s - &s_ref).norm() / s_ref.norm();
    assert!(
        s_err < 1e-9,
        "s was not rotated back onto s_ref (rel. err {s_err:.3e})"
    );

    // 2. THE SIGN: A must absorb the inverse rotation, so the product is invariant.
    //    With the sign flipped this is the dominant failure - the product is rotated by
    //    e^{2i*theta} per component instead of being left alone.
    let p_err = (&a * &s - &product_before).norm() / product_before.norm();
    assert!(
        p_err < 1e-9,
        "align_subject changed A*S (rel. err {p_err:.3e}); the A counter-rotation sign is wrong"
    );
}

/// The concat loop indexes every subject with `subjects[0].ncols()`, so ragged input would
/// panic out of bounds. It must be a clean Err instead.
#[test]
fn ragged_subjects_are_rejected() {
    let a = DMatrix::<Complex64>::zeros(4, 6);
    let b = DMatrix::<Complex64>::zeros(4, 5); // one voxel short
    let err = run_complex_ica(&[a, b], 2, None, Estimator::NcFastica, &fixtures_dir())
        .expect_err("ragged subjects must be rejected, not panic");
    assert!(err.contains("subject 1"), "unhelpful error: {err}");

    let err = run_complex_ica(&[], 2, None, Estimator::NcFastica, &fixtures_dir())
        .expect_err("no subjects must be rejected");
    assert!(err.contains("no subjects"), "unhelpful error: {err}");
}
