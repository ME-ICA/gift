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

pub struct PipelineResult {
    pub s_group: DMatrix<Complex64>,
    pub a_group: DMatrix<Complex64>,
    pub mask: Vec<bool>,
    pub subjects: Vec<(DMatrix<Complex64>, DMatrix<Complex64>)>,
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
    let (mask, _q, _tau) = phase_quality_mask(&z, None);

    // 2. restrict every subject to the masked voxels
    let kept: Vec<usize> = (0..v).filter(|&j| mask[j]).collect();
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
        let theta = align_to_reference(&mut s_i, &s_group);
        for (k, &th) in theta.iter().enumerate() {
            let inv = Complex64::new(th.cos(), -th.sin());
            for r in 0..a_i.nrows() {
                a_i[(r, k)] *= inv;
            }
        }
        out.push((s_i, a_i));
    }

    Ok(PipelineResult { s_group, a_group, mask, subjects: out })
}
