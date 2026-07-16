//! Group complex ICA: two-stage PCA reduction and GICA back-reconstruction.
//!
//! Follows GIFT's standard scheme: reduce each subject, temporally concatenate, reduce
//! again at the group level, run ICA on the group-reduced data, then back-reconstruct
//! per-subject maps and time courses.

use nalgebra::DMatrix;
use num_complex::Complex64;

use crate::whiten::whiten_hermitian;

pub struct Reduction {
    pub xg: DMatrix<Complex64>,             // (n_group, V)
    pub reduced: Vec<DMatrix<Complex64>>,   // per subject: (n_subject, V)
    pub whiteners: Vec<DMatrix<Complex64>>, // per subject: (n_subject, T_i)
    pub w_group: DMatrix<Complex64>,        // (n_group, n_sub * n_subject)
}

/// Subject-level then group-level complex PCA. Each subject is (T_i, V), already masked.
pub fn two_stage_pca(
    subjects: &[DMatrix<Complex64>],
    n_subject: usize,
    n_group: usize,
) -> Result<Reduction, String> {
    if subjects.is_empty() {
        return Err("two_stage_pca: no subjects".into());
    }

    let mut reduced = Vec::with_capacity(subjects.len());
    let mut whiteners = Vec::with_capacity(subjects.len());
    for xi in subjects {
        // Remove the per-VOXEL TEMPORAL mean: strips the static complex baseline image,
        // which is constant over time and would otherwise be a huge rank-1 nuisance
        // direction that eats a PCA slot and silently discards a real source.
        //
        // whiten_hermitian separately removes the mean along ITS sample axis - a
        // different operation. Both are needed. This must run AFTER the phase mask.
        let mut x = xi.clone();
        let t = x.nrows();
        for mut col in x.column_iter_mut() {
            let mean: Complex64 = col.iter().sum::<Complex64>() / Complex64::new(t as f64, 0.0);
            col.iter_mut().for_each(|z| *z -= mean);
        }

        let w = whiten_hermitian(&x, n_subject)?;
        reduced.push(w.xw);
        whiteners.push(w.w_whiten);
    }

    // temporal concatenation of the subject-reduced data
    let v = reduced[0].ncols();
    let rows: usize = reduced.iter().map(|r| r.nrows()).sum();
    let mut stacked = DMatrix::<Complex64>::zeros(rows, v);
    let mut offset = 0;
    for r in &reduced {
        stacked.view_mut((offset, 0), (r.nrows(), v)).copy_from(r);
        offset += r.nrows();
    }

    let g = whiten_hermitian(&stacked, n_group)?;
    Ok(Reduction {
        xg: g.xw,
        reduced,
        whiteners,
        w_group: g.w_whiten,
    })
}

/// GICA back-reconstruction to subject-specific maps and time courses.
///
/// The group model is Xg = A_group * S_group in the group-reduced space. Undo the group
/// whitener to express the group mixing in the STACKED subject-reduced space:
///
/// ```text
/// B = pinv(W_group) * A_group        // (n_sub * n_subject, N)
/// ```
///
/// and partition B by subject into Bi (n_subject, N). Bi maps the sources into subject i's
/// reduced space, so subject i's own maps come from projecting that subject's ACTUAL
/// reduced data through it:
///
/// ```text
/// S_i = pinv(Bi) * Y_i               // (N, V)
/// ```
///
/// Using Y_i (the real data) is the whole point: `pinv(Bi) * (Bi * S_group)` would collapse
/// to S_group for every subject and carry zero subject-specific variance.
pub fn back_reconstruct(
    a_group: &DMatrix<Complex64>,
    red: &Reduction,
) -> Vec<(DMatrix<Complex64>, DMatrix<Complex64>)> {
    let pinv_wg = red
        .w_group
        .clone()
        .pseudo_inverse(1e-12)
        .expect("pinv of the group whitener");
    let b = pinv_wg * a_group; // (n_sub * n_subject, N)

    let mut out = Vec::with_capacity(red.reduced.len());
    let mut offset = 0usize;
    for (y_i, w_i) in red.reduced.iter().zip(red.whiteners.iter()) {
        let rows = y_i.nrows();
        let bi = b.view((offset, 0), (rows, b.ncols())).into_owned(); // (n_subject, N)
        offset += rows;

        let pinv_bi = bi
            .clone()
            .pseudo_inverse(1e-12)
            .expect("pinv of the subject block");
        let s_i = pinv_bi * y_i; // (N, V) - uses the subject's ACTUAL reduced data

        let pinv_wi = w_i
            .clone()
            .pseudo_inverse(1e-12)
            .expect("pinv of the subject whitener");
        let a_i = pinv_wi * &bi; // (T_i, N)

        out.push((s_i, a_i));
    }
    out
}
