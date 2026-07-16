//! Complex ICA-EBM: complex ICA by entropy bound minimization.
//!
//! Ported from GroupICAT/icatb/icatb_analysis_functions/icatb_algorithms/complex_ica/
//! complex_ICA_EBM.m (functions `CEBM`, `complex_ICA_EBM_north`, `pre_processing` and
//! `inv_sqrtmH`), by way of the validated Python translation at
//! python/complex_gift/estimators/cebm.py. The Python is the authority: it was validated
//! against MATLAB on separation quality (ISI 0.0055 vs MATLAB 0.0075 across 15 seeds).
//! Where Python and MATLAB appear to differ, this port follows the Python.
//!
//! Reference:
//!     Xi-Lin Li and Tulay Adali, "Complex independent component analysis by entropy
//!     bound minimization," IEEE Trans. Circuits and Systems I, 57(7):1417-1430,
//!     July 2010.
//!
//! Copyright (C) 2023 MLSP Lab (original MATLAB); this Rust translation.
//!
//! This program is free software: you can redistribute it and/or modify it under
//! the terms of the GNU General Public License as published by the Free Software
//! Foundation, either version 3 of the License, or (at your option) any later
//! version.
//!
//! This program is distributed in the hope that it will be useful, but WITHOUT
//! ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
//! FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
//! details: <https://www.gnu.org/licenses/>.
//!
//! ## Deliberate deviations from the MATLAB reference (inherited from the Python port)
//! * **Injected, seeded RNG.** The reference draws from MATLAB's *global* RNG state, so its
//!   runs are not reproducible without touching global state. Here every draw comes from one
//!   seeded PRNG, so a run is reproducible given a seed. Consequently this port cannot - and
//!   is not meant to - reproduce MATLAB's `W` elementwise; it is validated on separation
//!   quality (ISI against the known true mixing).
//! * **`type2` (the second kind of entropy bound) is not ported.** `CEBM` calls the optimizer
//!   with `type2 = 0`, so the whole `if type2` branch is unreachable from the only entry
//!   point (the caller that would set `type2 = 1` is commented out at complex_ICA_EBM.m:241).

use std::path::Path;

use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;

use super::EstimatorResult;
use crate::nf_table::{load_nf_table, simplified_ppval, Nf};

/// Minimal reproducible PRNG. CEBM is stochastic; a Rust run must be repeatable given a
/// seed. It does NOT need to match Python's random stream - only to be deterministic.
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Rng(seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407)
            | 1)
    }
    /// uniform in [0, 1)
    pub(crate) fn next_f64(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 11) as f64 / (1u64 << 53) as f64
    }
    /// standard normal (Box-Muller)
    pub(crate) fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(f64::MIN_POSITIVE);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

// The reference allocates 8 nonlinearities but only ever fills/uses 1, 3, 5, 7 (1-based).
// Entries 2, 4, 6, 8 stay at zero and still take part in the max(), so a zero bound (the
// Gaussian case) can win. Kept faithfully.
const K_REAL: usize = 8;

#[inline]
fn cr(re: f64) -> Complex64 {
    Complex64::new(re, 0.0)
}

/// Pseudo-covariance E[x x^T] - PLAIN transpose, NOT the conjugate transpose.
///
/// This is the quantity that carries the noncircularity CEBM exploits. Using `.adjoint()`
/// here silently destroys it while still appearing to work - and the ISI oracle test
/// CANNOT detect the swap (proven in Phase 2), which is why `pseudo_cov` is a named,
/// separately-guarded function rather than an inline expression.
pub fn pseudo_cov(x: &DMatrix<Complex64>) -> DMatrix<Complex64> {
    let t = x.ncols() as f64;
    (x * x.transpose()) / cr(t)
}

/// Inverse matrix square root of a Hermitian matrix (MATLAB `inv_sqrtmH`).
pub(crate) fn inv_sqrtm_h(b: &DMatrix<Complex64>) -> DMatrix<Complex64> {
    let eig = b.clone().symmetric_eigen();
    let n = b.nrows();
    let mut out = DMatrix::<Complex64>::zeros(n, n);
    for k in 0..n {
        let s = cr(1.0 / eig.eigenvalues[k].sqrt());
        for i in 0..n {
            for j in 0..n {
                out[(i, j)] += s * eig.eigenvectors[(i, k)] * eig.eigenvectors[(j, k)].conj();
            }
        }
    }
    out
}

/// MATLAB `pre_processing`: remove DC, then whiten by inv_sqrtmH(X X^H / T).
/// Returns (whitened X, the whitener P).
pub(crate) fn pre_processing(x: &DMatrix<Complex64>) -> (DMatrix<Complex64>, DMatrix<Complex64>) {
    let t = x.ncols();
    let mut xc = x.clone();
    for mut row in xc.row_iter_mut() {
        let mean: Complex64 = row.iter().sum::<Complex64>() / cr(t as f64);
        row.iter_mut().for_each(|z| *z -= mean);
    }
    // conjugate transpose here: this is the COVARIANCE, not the pseudo-covariance
    let r = (&xc * xc.adjoint()) / cr(t as f64);
    let p = inv_sqrtm_h(&r);
    let xw = &p * &xc;
    (xw, p)
}

/// MATLAB `inv(sqrtm(W*W'))*W`. `W W^H` is Hermitian, so the eig decomposition is exact.
fn sym_decorrelate(w: &DMatrix<Complex64>) -> DMatrix<Complex64> {
    inv_sqrtm_h(&(w * w.adjoint())) * w
}

/// Remove a single row from a matrix, preserving the order of the rest.
fn remove_row(m: &DMatrix<Complex64>, r: usize) -> DMatrix<Complex64> {
    let n = m.nrows();
    let cols = m.ncols();
    let mut out = DMatrix::<Complex64>::zeros(n - 1, cols);
    let mut oi = 0;
    for i in 0..n {
        if i == r {
            continue;
        }
        out.set_row(oi, &m.row(i));
        oi += 1;
    }
    out
}

/// numpy `np.sign`: sign(0) == 0 (Rust's `f64::signum` returns +/-1 for zeros, which differs).
#[inline]
fn npsign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Plain (non-conjugated) dot product: sum a_i * b_i.
#[inline]
fn cdot(a: &DVector<Complex64>, b: &DVector<Complex64>) -> Complex64 {
    let mut s = Complex64::new(0.0, 0.0);
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

/// Hermitian dot product: sum conj(a_i) * b_i.
#[inline]
fn hdot(a: &DVector<Complex64>, b: &DVector<Complex64>) -> Complex64 {
    let mut s = Complex64::new(0.0, 0.0);
    for i in 0..a.len() {
        s += a[i].conj() * b[i];
    }
    s
}

/// N normals cast to complex (imag = 0), matching `rng.standard_normal(N).astype(complex128)`.
fn randn_real_vec(rng: &mut Rng, n: usize) -> DVector<Complex64> {
    let vals: Vec<Complex64> = (0..n).map(|_| cr(rng.next_normal())).collect();
    DVector::from_vec(vals)
}

/// `randn(N,M) + 1j*randn(N,M)`: draw all reals, then all imags (row-major).
fn randn_complex_matrix(rng: &mut Rng, n: usize, m: usize) -> DMatrix<Complex64> {
    let re: Vec<f64> = (0..n * m).map(|_| rng.next_normal()).collect();
    let im: Vec<f64> = (0..n * m).map(|_| rng.next_normal()).collect();
    DMatrix::from_fn(n, m, |i, j| Complex64::new(re[i * m + j], im[i * m + j]))
}

/// The reference's per-component whitening of the real/imaginary parts.
struct Stand {
    z_real: Vec<f64>,
    z_imag: Vec<f64>,
    sigma_r2: f64,
    sigma_i2: f64,
    sigma_r: f64,
    rho: f64,
    delta1: f64,
    u: Vec<f64>,
    v: Vec<f64>,
}

fn standardize(z: &DVector<Complex64>, t: usize) -> Stand {
    let tf = t as f64;
    let z_real: Vec<f64> = z.iter().map(|c| c.re).collect();
    let z_imag: Vec<f64> = z.iter().map(|c| c.im).collect();
    let sigma_r2 = z_real.iter().map(|r| r * r).sum::<f64>() / tf;
    let sigma_i2 = z_imag.iter().map(|r| r * r).sum::<f64>() / tf;
    let sigma_r = sigma_r2.sqrt();
    let rho = z_real.iter().zip(&z_imag).map(|(r, i)| r * i).sum::<f64>() / tf;
    let delta1 = sigma_r2 * sigma_i2 - rho * rho;
    let sqrt_d1 = delta1.sqrt();
    let u: Vec<f64> = z_real.iter().map(|r| r / sigma_r).collect();
    let v: Vec<f64> = z_real
        .iter()
        .zip(&z_imag)
        .map(|(r, i)| sigma_r * i / sqrt_d1 - rho * r / sigma_r / sqrt_d1)
        .collect();
    Stand {
        z_real,
        z_imag,
        sigma_r2,
        sigma_i2,
        sigma_r,
        rho,
        delta1,
        u,
        v,
    }
}

/// Negentropy bound for G1 = x^4: clamp the argument, no slope extrapolation.
fn ne_bound_g1(nf: &Nf, eg: f64) -> f64 {
    if eg < nf.min_egx {
        simplified_ppval(&nf.pp, nf.min_egx)
    } else if eg > nf.max_egx {
        simplified_ppval(&nf.pp, nf.max_egx)
    } else {
        simplified_ppval(&nf.pp, eg)
    }
}

/// Negentropy bound for G3/G5/G7: linear (abs-valued) extrapolation outside the table.
fn ne_bound_slope(nf: &Nf, eg: f64) -> f64 {
    if eg < nf.min_egx {
        let d = simplified_ppval(&nf.pp_slope, nf.min_egx) * (eg - nf.min_egx);
        simplified_ppval(&nf.pp, nf.min_egx) + d.abs()
    } else if eg > nf.max_egx {
        let d = simplified_ppval(&nf.pp_slope, nf.max_egx) * (eg - nf.max_egx);
        simplified_ppval(&nf.pp, nf.max_egx) + d.abs()
    } else {
        simplified_ppval(&nf.pp, eg)
    }
}

/// MATLAB `min(max(EG, min_EGx), max_EGx)`.
fn clip(nf: &Nf, value: f64) -> f64 {
    value.max(nf.min_egx).min(nf.max_egx)
}

/// NE_Bound / EG vectors (length 8, 1-based semantics) plus xx/sign/abs for one real signal.
struct Bounds {
    ne: [f64; K_REAL],
    eg: [f64; K_REAL],
    xx: Vec<f64>,
    sign: Vec<f64>,
    abs: Vec<f64>,
}

fn bounds(nf: &[Nf; 8], x: &[f64], t: usize) -> Bounds {
    let tf = t as f64;
    let xx: Vec<f64> = x.iter().map(|v| v * v).collect();
    let sign: Vec<f64> = x.iter().map(|&v| npsign(v)).collect();
    let abs: Vec<f64> = sign.iter().zip(x).map(|(s, v)| s * v).collect();

    let mut ne = [0.0f64; K_REAL];
    let mut eg = [0.0f64; K_REAL];

    // G1 = x^4  (nf1 -> nf[0])
    eg[0] = xx.iter().map(|v| v * v).sum::<f64>() / tf;
    ne[0] = ne_bound_g1(&nf[0], eg[0]);
    // G3 = |x|/(1+|x|)  (nf3 -> nf[2])
    eg[2] = abs.iter().map(|a| a / (1.0 + a)).sum::<f64>() / tf;
    ne[2] = ne_bound_slope(&nf[2], eg[2]);
    // G5 = x*|x|/(10+|x|)  (nf5 -> nf[4])
    eg[4] = x
        .iter()
        .zip(&abs)
        .map(|(xv, a)| xv * a / (10.0 + a))
        .sum::<f64>()
        / tf;
    ne[4] = ne_bound_slope(&nf[4], eg[4]);
    // G7 = x/(1+x^2)  (nf7 -> nf[6])
    eg[6] = x
        .iter()
        .zip(&xx)
        .map(|(xv, xxv)| xv / (1.0 + xxv))
        .sum::<f64>()
        / tf;
    ne[6] = ne_bound_slope(&nf[6], eg[6]);

    Bounds {
        ne,
        eg,
        xx,
        sign,
        abs,
    }
}

/// First-max argmax (numpy semantics: ties resolve to the lowest index).
fn argmax(a: &[f64]) -> usize {
    let mut best = 0usize;
    for i in 1..a.len() {
        if a[i] > a[best] {
            best = i;
        }
    }
    best
}

/// The "sea" fixed-point algorithm that provides the initial guess (CEBM:55-221).
fn sea(xc: &DMatrix<Complex64>, nf: &[Nf; 8], rng: &mut Rng, tolerance: f64) -> DMatrix<Complex64> {
    let maxiter_sea = 100usize;
    let max_cost_increase_number = 10usize;
    let (n, t) = xc.shape();
    let tf = t as f64;

    let w0 = randn_complex_matrix(rng, n, n);
    let mut w = sym_decorrelate(&w0);
    let mut last_w = w.clone();
    let mut best_w = w.clone();

    // PLAIN transpose: C is the PSEUDO-covariance (CEBM:59). Not the covariance.
    let cmat = pseudo_cov(xc);

    let mut min_cost = f64::INFINITY;
    let mut cost_increase_counter = 0usize;

    for _ in 0..maxiter_sea {
        let mut cost = 0.0;

        for row in 0..n {
            // vec = W(n,:).'  (the row, as-is)
            let vec: DVector<Complex64> = w.row(row).transpose();
            // y = v.'*Xc : y[k] = sum_i vec[i]*Xc[i,k]
            let y: DVector<Complex64> = xc.transpose() * &vec;
            // Cv = C*vec
            let cv = &cmat * &vec;
            // v = conj(Xc)*(y.*y.*conj(y)).'/T - 2*v - (v.'*C*v)*conj(C*v)
            let yyy: DVector<Complex64> =
                DVector::from_iterator(t, y.iter().map(|yv| yv * yv * yv.conj()));
            let term1 = (xc.conjugate() * yyy) / cr(tf);
            let vcv = cdot(&vec, &cv); // v.'*C*v  (plain)
            let new_vec = term1 - &vec * cr(2.0) - cv.conjugate() * vcv;
            w.set_row(row, &new_vec.transpose());

            // evaluate the cost with the PRE-update row (z = y)
            let st = standardize(&y, t);
            cost += 0.5 * st.delta1.ln() + (2.0 * std::f64::consts::PI).ln() + 1.0;
            let bu = bounds(nf, &st.u, t);
            let bv = bounds(nf, &st.v, t);
            let max_u = bu.ne.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let max_v = bv.ne.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            cost -= max_u + max_v;
        }

        if cost < min_cost {
            min_cost = cost;
            cost_increase_counter = 0;
            best_w = last_w.clone(); // the W this cost was computed from
        } else {
            cost_increase_counter += 1;
        }

        w = sym_decorrelate(&w);
        if cost_increase_counter > max_cost_increase_number {
            break;
        }
        // 1 - min |diag(W * last_W^H)| < tol
        let mut min_diag = f64::INFINITY;
        for i in 0..n {
            let mut d = Complex64::new(0.0, 0.0);
            for k in 0..n {
                d += w[(i, k)] * last_w[(i, k)].conj();
            }
            min_diag = min_diag.min(d.norm());
        }
        if 1.0 - min_diag < tolerance {
            break;
        }
        last_w = w.clone();
    }

    best_w
}

/// `complex_ICA_EBM_north`: the nonorthogonal ICA optimizer (CEBM:269-711).
///
/// `type2` is fixed at 0; the second-kind entropy bound is unreachable from `CEBM` and is
/// not ported. `stochastic_search` is always true (the only caller sets it).
#[allow(clippy::too_many_arguments)]
fn north(
    x: &DMatrix<Complex64>,
    w0: &DMatrix<Complex64>,
    max_iter_north: usize,
    mu0_north: f64,
    max_cost_increase_number: usize,
    nf: &[Nf; 8],
    rng: &mut Rng,
) -> Result<DMatrix<Complex64>, String> {
    let (n, t) = x.shape();
    let r_xxt = pseudo_cov(x); // PLAIN transpose: pseudo-covariance of the whitened data

    let mut cost_increase_counter = 0usize;
    let mut mu = mu0_north;
    let mut w = w0.clone();
    let mut best_w = w.clone();
    let mut last_w = w.clone();
    let mut min_cost = f64::INFINITY;
    let mut max_negentropy = vec![0.0f64; n];
    let mut negentropy_array = vec![0.0f64; n];

    // MATLAB's `grad` persists across rows/iterations. If `p0` matches no case (possible when
    // every NE_Bound is negative, so an untouched zero at an even index wins the max), MATLAB
    // silently reuses the previous row's `grad`. Reproduced here.
    let mut grad: Option<DVector<Complex64>> = None;
    let mut inv_q: Option<DMatrix<Complex64>> = None;

    let mut iter_idx = 0usize;
    while iter_idx < max_iter_north {
        // cost = -log|det(W^H W)|
        let gram = w.adjoint() * &w;
        let mut cost = -(gram.determinant().norm().ln());

        // `row` drives branching logic (n > 7 / row == 0), not just indexing; the index
        // form mirrors the MATLAB reference this is transcribed from.
        #[allow(clippy::needless_range_loop)]
        for row in 0..n {
            let h: DVector<Complex64> = if n > 7 {
                if row == 0 {
                    let wn = remove_row(&w, 0);
                    let q = (&wn * wn.adjoint())
                        .try_inverse()
                        .ok_or_else(|| "cebm north: singular Gram matrix in inv_Q".to_string())?;
                    inv_q = Some(q);
                } else {
                    let n_last = row - 1;
                    let wn_last = remove_row(&w, n_last);
                    let w_current: DVector<Complex64> = w.row(row).transpose().conjugate();
                    let w_last: DVector<Complex64> = w.row(n_last).transpose().conjugate();
                    let mut c = &wn_last * (&w_last - &w_current);
                    c[n_last] = cr(0.5) * (hdot(&w_last, &w_last) - hdot(&w_current, &w_current));

                    let iq = inv_q.as_ref().unwrap();
                    let temp1 = iq * &c;
                    let temp2: DVector<Complex64> = iq.column(n_last).into_owned();
                    let inv_q_plus = iq
                        - (&temp1 * temp2.adjoint()) / (Complex64::new(1.0, 0.0) + temp1[n_last]);

                    let temp1b = inv_q_plus.adjoint() * &c;
                    let temp2b: DVector<Complex64> = inv_q_plus.column(n_last).into_owned();
                    let denom = Complex64::new(1.0, 0.0) + hdot(&c, &temp2b);
                    let mut iq_new = &inv_q_plus - (&temp2b * temp1b.adjoint()) / denom;
                    iq_new = (&iq_new + iq_new.adjoint()) / cr(2.0); // inv_Q is Hermitian
                    inv_q = Some(iq_new);
                }

                let temp1 = randn_real_vec(rng, n);
                let w_n = remove_row(&w, row);
                let iq = inv_q.as_ref().unwrap();
                &temp1 - w_n.adjoint() * (iq * (&w_n * &temp1))
            } else {
                let temp1 = randn_real_vec(rng, n);
                let temp2 = remove_row(&w, row);
                let m = &temp2 * temp2.adjoint();
                let m_inv = m
                    .try_inverse()
                    .ok_or_else(|| "cebm north: singular Gram matrix in h".to_string())?;
                let y = m_inv * (&temp2 * &temp1);
                &temp1 - temp2.adjoint() * y
            };

            // w = W(n,:)' (conjugate of the row); rn = W(n,:) (row as-is, = w.conj())
            let rn: DVector<Complex64> = w.row(row).transpose();
            let w_col: DVector<Complex64> = rn.conjugate();
            // z = w'*X == W(n,:) @ X : z[k] = sum_i rn[i]*X[i,k]
            let z: DVector<Complex64> = x.transpose() * &rn;

            let st = standardize(&z, t);
            let sqrt_d1 = st.delta1.sqrt();

            cost += 0.5 * st.delta1.ln();
            negentropy_array[row] = -0.5 * st.delta1.ln();

            let bu = bounds(nf, &st.u, t);
            let bv = bounds(nf, &st.v, t);
            let p0 = argmax(&bu.ne) + 1; // back to 1-based (the switch index)
            let q0 = argmax(&bv.ne) + 1;
            let bound_real = bu.ne[p0 - 1] + bv.ne[q0 - 1];

            cost -= bound_real;
            negentropy_array[row] += bound_real;

            // R_xxt @ w.conj()  (w.conj() == rn)
            let rw = &r_xxt * &rn;
            let grad_delta1 = &rw * Complex64::new(0.5 * (st.sigma_i2 - st.sigma_r2), st.rho);

            // --- real-part (u) weight draw happens unconditionally ---
            let weight: Vec<f64> = (0..t).map(|_| rng.next_f64()).collect();
            let sw: f64 = weight.iter().sum();

            let p_used = matches!(p0, 1 | 3 | 5 | 7);
            let mut g: DVector<Complex64> = if p_used {
                let (v_egu, gu): (f64, Vec<f64>) = match p0 {
                    1 => {
                        // G = x^4, g = 4x^3 (NOT clipped, per the reference)
                        let ve = simplified_ppval(&nf[0].pp_slope, bu.eg[0]);
                        let gu = (0..t).map(|k| 4.0 * bu.xx[k] * st.u[k]).collect();
                        (ve, gu)
                    }
                    3 => {
                        let ve = simplified_ppval(&nf[2].pp_slope, clip(&nf[2], bu.eg[2]));
                        let gu = (0..t)
                            .map(|k| bu.sign[k] / (1.0 + bu.abs[k]).powi(2))
                            .collect();
                        (ve, gu)
                    }
                    5 => {
                        let ve = simplified_ppval(&nf[4].pp_slope, clip(&nf[4], bu.eg[4]));
                        let gu = (0..t)
                            .map(|k| bu.abs[k] * (20.0 + bu.abs[k]) / (10.0 + bu.abs[k]).powi(2))
                            .collect();
                        (ve, gu)
                    }
                    _ => {
                        // p0 == 7; G = x/(1+x^2), g = (1-x^2)/(1+x^2)^2
                        let ve = simplified_ppval(&nf[6].pp_slope, clip(&nf[6], bu.eg[6]));
                        let gu = (0..t)
                            .map(|k| (1.0 - bu.xx[k]) / (1.0 + bu.xx[k]).powi(2))
                            .collect();
                        (ve, gu)
                    }
                };

                // grad = 0.5*grad_delta1/Delta1
                let mut base = &grad_delta1 * cr(0.5 / st.delta1);
                // grad -= vEGu * X*(weight.*gu)' / sw / 2 / sigma_R
                let wgu: DVector<Complex64> =
                    DVector::from_iterator(t, (0..t).map(|k| cr(weight[k] * gu[k])));
                base -= (x * wgu) * cr(v_egu / sw / 2.0 / st.sigma_r);
                // grad -= vEGu * sum(-weight.*gu.*z_real) / sw / 4 / sigma_R^3 * rw
                let s_real: f64 = (0..t).map(|k| -weight[k] * gu[k] * st.z_real[k]).sum();
                base -= &rw * cr(v_egu * s_real / sw / 4.0 / st.sigma_r.powi(3));
                base
            } else {
                match grad {
                    Some(ref g_prev) => g_prev.clone(),
                    None => {
                        return Err(
                            "complex ICA-EBM: every real-part negentropy bound was negative on \
                             the first component, so no gradient is defined (MATLAB errors here \
                             too). The data may be degenerate."
                                .to_string(),
                        )
                    }
                }
            };

            // --- imaginary-part (v) weight draw happens unconditionally ---
            let weight: Vec<f64> = (0..t).map(|_| rng.next_f64()).collect();
            let sw: f64 = weight.iter().sum();

            if matches!(q0, 1 | 3 | 5 | 7) {
                let (v_egv, gv, gv_v): (f64, Vec<f64>, Vec<f64>) = match q0 {
                    1 => {
                        let ve = simplified_ppval(&nf[0].pp_slope, bv.eg[0]);
                        let gv: Vec<f64> = (0..t).map(|k| 4.0 * bv.xx[k] * st.v[k]).collect();
                        let gv_v: Vec<f64> = (0..t).map(|k| 4.0 * bv.xx[k] * bv.xx[k]).collect();
                        (ve, gv, gv_v)
                    }
                    3 => {
                        let ve = simplified_ppval(&nf[2].pp_slope, clip(&nf[2], bv.eg[2]));
                        let gv: Vec<f64> = (0..t)
                            .map(|k| bv.sign[k] / (1.0 + bv.abs[k]).powi(2))
                            .collect();
                        let gv_v: Vec<f64> = (0..t)
                            .map(|k| bv.abs[k] / (1.0 + bv.abs[k]).powi(2))
                            .collect();
                        (ve, gv, gv_v)
                    }
                    5 => {
                        let ve = simplified_ppval(&nf[4].pp_slope, clip(&nf[4], bv.eg[4]));
                        let gv: Vec<f64> = (0..t)
                            .map(|k| bv.abs[k] * (20.0 + bv.abs[k]) / (10.0 + bv.abs[k]).powi(2))
                            .collect();
                        let gv_v: Vec<f64> = (0..t).map(|k| gv[k] * st.v[k]).collect();
                        (ve, gv, gv_v)
                    }
                    _ => {
                        // q0 == 7
                        let ve = simplified_ppval(&nf[6].pp_slope, clip(&nf[6], bv.eg[6]));
                        let gv: Vec<f64> = (0..t)
                            .map(|k| (1.0 - bv.xx[k]) / (1.0 + bv.xx[k]).powi(2))
                            .collect();
                        let gv_v: Vec<f64> = (0..t).map(|k| gv[k] * st.v[k]).collect();
                        (ve, gv, gv_v)
                    }
                };

                // grad += vEGv * X*(weight.*gv)' / sw/2/sigma_R/sqrt_D1 * (rho + i*sigma_R2)
                let wgv: DVector<Complex64> =
                    DVector::from_iterator(t, (0..t).map(|k| cr(weight[k] * gv[k])));
                let coef_a = Complex64::new(st.rho, st.sigma_r2)
                    * cr(v_egv / sw / 2.0 / st.sigma_r / sqrt_d1);
                g += (x * wgv) * coef_a;

                // grad += vEGv * sum(weight.*gv_v) / sw/2/Delta1 * grad_delta1
                let s2: f64 = (0..t).map(|k| weight[k] * gv_v[k]).sum();
                g += &grad_delta1 * cr(v_egv * s2 / sw / 2.0 / st.delta1);

                // grad -= vEGv * sum(weight.*gv.*(z_imag + (rho/sigma_R2 + 2i)*z_real))
                //                / sw/4/sigma_R/sqrt_D1 * rw
                let rho_coef = Complex64::new(st.rho / st.sigma_r2, 2.0);
                let mut s3 = Complex64::new(0.0, 0.0);
                for k in 0..t {
                    let bracket = cr(st.z_imag[k]) + rho_coef * st.z_real[k];
                    s3 += cr(weight[k] * gv[k]) * bracket;
                }
                let coef_c = s3 * cr(v_egv / sw / 4.0 / st.sigma_r / sqrt_d1);
                g -= &rw * coef_c;
            }

            // grad = grad - h/(w'*h)
            g -= &h * (Complex64::new(1.0, 0.0) / cdot(&rn, &h));
            // grad = grad - real(w'*grad)*w
            let proj = cdot(&rn, &g).re;
            g -= &w_col * cr(proj);
            // grad = grad / norm(grad)
            let gn = g.norm();
            g /= cr(gn);

            grad = Some(g.clone()); // persists for possible reuse next row

            // w1 = w - mu*grad; w1 = w1/norm(w1); W(n,:) = w1'
            let mut w1 = &w_col - &g * cr(mu);
            let w1n = w1.norm();
            w1 /= cr(w1n);
            w.set_row(row, &w1.conjugate().transpose());
        }

        if cost < min_cost {
            min_cost = cost;
            best_w = last_w.clone(); // the W this cost was computed from
            max_negentropy = negentropy_array.clone();
            cost_increase_counter = 0;
        } else {
            cost_increase_counter += 1;
        }

        iter_idx += 1;

        if cost_increase_counter > max_cost_increase_number {
            let mu_floor = 1.0 / 20.0; // stochastic_search branch
            if mu > mu_floor {
                mu /= 2.0;
                cost_increase_counter = 0;
                w = best_w.clone();
                last_w = w.clone();
                continue;
            }
            break;
        }

        last_w = w.clone();
    }

    let w_best = best_w;

    // sort the components by negentropy, descending (stable)
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        max_negentropy[b]
            .partial_cmp(&max_negentropy[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut out = DMatrix::<Complex64>::zeros(n, n);
    for (new_i, &old_i) in idx.iter().enumerate() {
        out.set_row(new_i, &w_best.row(old_i));
    }
    Ok(out)
}

/// Complex ICA by entropy bound minimization (MATLAB `CEBM`).
///
/// `x`: (N, T) complex mixtures. `nf_dir`: directory of the nonlinearity `.npy` tables.
/// `seed`: drives every random draw so a run is reproducible. `tol`: SEA stopping tolerance
/// (MATLAB `tolerance`, 1e-4). `max_iter`: max iterations of the nonorthogonal optimizer
/// (defaults to the reference's 1000).
///
/// Returns `EstimatorResult` where `w` demixes the ORIGINAL `x` (the pre-whitener `P` is
/// folded in, as in `CEBM`: `W = W*P`), `a = pinv(w)`, `s = w*x`.
pub fn cebm(
    x: &DMatrix<Complex64>,
    nf_dir: &Path,
    seed: u64,
    tol: f64,
    max_iter: Option<usize>,
) -> Result<EstimatorResult, String> {
    if x.nrows() == 0 || x.ncols() == 0 {
        return Err("cebm: X must be a non-empty (N, T) matrix".to_string());
    }
    let mut rng = Rng::new(seed);
    let nf = load_nf_table(nf_dir);
    let max_iter_north = max_iter.unwrap_or(1000);

    let (xc, p) = pre_processing(x);

    // initial guess
    let w_init = sea(&xc, &nf, &mut rng, tol);

    // stochastic gradient search. The reference's "refinement" pass (complex_ICA_EBM.m:241)
    // is commented out there, so it is not run here either.
    let w_north = north(&xc, &w_init, max_iter_north, 1.0 / 5.0, 5, &nf, &mut rng)?;

    // fold the pre-whitener back in: W now demixes the original X
    let w = w_north * p;

    let a = w
        .clone()
        .pseudo_inverse(1e-12)
        .map_err(|e| format!("cebm: pseudo-inverse of W failed: {e}"))?;
    let s = &w * x;

    Ok(EstimatorResult { w, a, s })
}
