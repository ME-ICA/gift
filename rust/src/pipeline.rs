//! The complete complex ICA pipeline, wired end to end.

use std::path::Path;

use nalgebra::DMatrix;
use num_complex::Complex64;

use crate::estimators::cebm::cebm;
use crate::estimators::nc_fastica::nc_fastica;
use crate::estimators::EstimatorResult;
use crate::group::{back_reconstruct, two_stage_pca};
use crate::phase_correct::{align_to_reference, correct_phase};
use crate::phase_mask::phase_quality_mask;

#[derive(Debug, Clone, Copy)]
pub enum Estimator {
    /// Complex ICA-EBM. Stochastic: the seed makes a run reproducible.
    Cebm { seed: u64 },
    /// Noncircular complex FastICA. Deterministic.
    NcFastica,
}

#[derive(Debug)]
pub struct PipelineResult {
    pub s_group: DMatrix<Complex64>,
    pub a_group: DMatrix<Complex64>,
    pub mask: Vec<bool>,
    pub subjects: Vec<(DMatrix<Complex64>, DMatrix<Complex64>)>,
}

/// Rotate one subject's maps into the group frame, counter-rotating `a` so the pair still
/// satisfies `x ~= a * s`. Returns the per-component rotation that was applied to `s`.
///
/// `align_to_reference` rotates row `k` of `s` by `e^{+i*theta_k}`, so column `k` of `a`
/// must take `e^{-i*theta_k}` for the product to be invariant.
///
/// Split out of `run_complex_ica` so the sign is testable: when the group model holds and
/// each subject block is full column rank, back-reconstructed maps already arrive in the
/// group frame, so `theta` is identically zero on realistic data and no end-to-end fixture
/// can tell the two signs apart. See `align_subject_*` in `tests/pipeline.rs`.
pub fn align_subject(
    s: &mut DMatrix<Complex64>,
    a: &mut DMatrix<Complex64>,
    s_ref: &DMatrix<Complex64>,
) -> Vec<f64> {
    let theta = align_to_reference(s, s_ref);
    for (k, &th) in theta.iter().enumerate() {
        let inv = Complex64::new(th.cos(), -th.sin());
        for r in 0..a.nrows() {
            a[(r, k)] *= inv;
        }
    }
    theta
}

/// Run group complex ICA over subjects given as (T_i, V) complex matrices.
pub fn run_complex_ica(
    subjects: &[DMatrix<Complex64>],
    n_components: usize,
    n_subject: Option<usize>,
    estimator: Estimator,
    nf_dir: &Path,
) -> Result<PipelineResult, String> {
    if subjects.is_empty() {
        return Err("run_complex_ica: no subjects".into());
    }
    let n_subject = n_subject.unwrap_or(n_components);
    let v = subjects[0].ncols();
    if let Some(i) = subjects.iter().position(|s| s.ncols() != v) {
        return Err(format!(
            "run_complex_ica: subject {i} has {} voxels, expected {v}",
            subjects[i].ncols()
        ));
    }

    // 1. phase-quality mask on the RAW concatenated data (V, T_total). This must precede
    //    any de-meaning: the mask keys on phase stability, which the static baseline
    //    provides.
    let t_total: usize = subjects.iter().map(|s| s.nrows()).sum();
    let mut z = DMatrix::<Complex64>::zeros(v, t_total);
    let mut off = 0usize;
    for s in subjects {
        for tt in 0..s.nrows() {
            for vv in 0..v {
                z[(vv, off + tt)] = s[(tt, vv)];
            }
        }
        off += s.nrows();
    }
    let (mask, q, _tau) = phase_quality_mask(&z, None);

    // Otsu needs two modes to separate. If Q has no meaningful spread, the threshold falls
    // on floating-point noise and the mask it returns is arbitrary - it will discard
    // genuine signal voxels while looking entirely plausible. Refuse instead, matching
    // Python, whose skimage.threshold_otsu raises ("Too many bins for data range") on the
    // same input. Silently inventing a mask is the worse failure mode, and no array-level
    // test downstream can see it.
    let q_lo = q.iter().copied().fold(f64::INFINITY, f64::min);
    let q_hi = q.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    // DELIBERATE `!(a > b)` rather than `a <= b`: NaN-safe, so an all-NaN Q is rejected
    // here too rather than sliding through. See whiten.rs for the full reasoning.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(q_hi - q_lo > 1e-9 * q_hi.abs().max(1.0)) {
        return Err(format!(
            "run_complex_ica: the phase-quality map is degenerate (Q spans [{q_lo:.6e}, \
             {q_hi:.6e}]); Otsu cannot separate a mask from it and would threshold on \
             numerical noise. Every voxel has effectively the same phase stability."
        ));
    }

    // 2. restrict every subject to the masked voxels
    let kept: Vec<usize> = (0..v).filter(|&j| mask[j]).collect();
    // An empty (or near-empty) mask is a data problem, not a bug, and this function's
    // contract is `Result` - it must not abort the caller. Without this guard, degenerate
    // input (e.g. an exactly constant phase-quality map, which drives Otsu's threshold to
    // the single Q value so nothing exceeds it) reaches whiten_hermitian and panics.
    if kept.len() < n_components {
        return Err(format!(
            "run_complex_ica: the phase-quality mask kept {} of {v} voxels, fewer than the \
             {n_components} components requested. The data may have no phase-stable voxels, \
             or the quality map may be degenerate (constant Q).",
            kept.len()
        ));
    }
    let masked: Vec<DMatrix<Complex64>> = subjects
        .iter()
        .map(|s| {
            let mut m = DMatrix::<Complex64>::zeros(s.nrows(), kept.len());
            for (c, &j) in kept.iter().enumerate() {
                for tt in 0..s.nrows() {
                    m[(tt, c)] = s[(tt, j)];
                }
            }
            m
        })
        .collect();

    // 3. two-stage complex PCA (removes the per-voxel temporal mean internally)
    let red = two_stage_pca(&masked, n_subject, n_components)?;

    // 4. complex ICA
    let res: EstimatorResult = match estimator {
        Estimator::Cebm { seed } => cebm(&red.xg, nf_dir, seed, 1e-4, None)?,
        Estimator::NcFastica => nc_fastica(&red.xg, "log", 1e-5, None)?,
    };

    // 5. phase-ambiguity correction on the group maps
    let mut s_group = res.s.clone();
    let mut a_group = res.a.clone();
    let _ = correct_phase(&mut s_group, &mut a_group, None);

    // 6. back-reconstruct per subject, align to the group, and counter-rotate A_i so the
    //    returned pair still satisfies X_i ~= A_i * S_i.
    let mut out = Vec::new();
    for (mut s_i, mut a_i) in back_reconstruct(&a_group, &red) {
        align_subject(&mut s_i, &mut a_i, &s_group);
        out.push((s_i, a_i));
    }

    Ok(PipelineResult {
        s_group,
        a_group,
        mask,
        subjects: out,
    })
}
