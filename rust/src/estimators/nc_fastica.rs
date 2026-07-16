//! Noncircular complex FastICA (symmetric orthogonalization).
//!
//! Ported from GroupICAT/icatb/icatb_analysis_functions/icatb_algorithms/complex_ica/
//! nonCircComplexFastICAsym.m, by way of the validated Python translation at
//! python/complex_gift/estimators/nc_fastica.py.
//!
//! Reference:
//!     Mike Novey and T. Adali, "On Extending the complex FastICA algorithm to
//!     noncircular sources," IEEE Trans. Signal Processing, 56(5):2148-2154, May 2008.
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

use nalgebra::DMatrix;
use num_complex::Complex64;

use super::cebm::pseudo_cov;
use super::EstimatorResult;

/// Nonlinearity smoothing constant (reference: `a2`). Live constant - NOT dead code
/// (unlike `tol`/`maxcounter` locals in the MATLAB original).
const A2: f64 = 0.05;

enum NonLinearity {
    Log,
    Kurt,
    Sqrt,
}

impl NonLinearity {
    fn parse(name: &str) -> Result<Self, String> {
        match name {
            "log" => Ok(NonLinearity::Log),
            "kurt" => Ok(NonLinearity::Kurt),
            "sqrt" => Ok(NonLinearity::Sqrt),
            other => Err(format!(
                "unknown nonlinearity {other:?}; expected one of \"log\", \"kurt\", \"sqrt\""
            )),
        }
    }

    /// g and its "derivative-like" companion gp, evaluated at `absy = |y|^2`.
    fn g_gp(&self, absy: f64) -> (f64, f64) {
        match self {
            NonLinearity::Log => (1.0 / (A2 + absy), -1.0 / (A2 + absy).powi(2)),
            NonLinearity::Kurt => (absy, 1.0),
            NonLinearity::Sqrt => (
                1.0 / (2.0 * (A2 + absy).sqrt()),
                -1.0 / (4.0 * (A2 + absy).powf(1.5)),
            ),
        }
    }
}

/// Frobenius norm of `|Wold^H W| - I`, the orthonormality-deviation convergence measure.
fn frob_diff(wold: &DMatrix<Complex64>, w: &DMatrix<Complex64>) -> f64 {
    let n = w.nrows();
    let m = wold.adjoint() * w;
    let mut acc = 0.0;
    for i in 0..n {
        for j in 0..n {
            let val = m[(i, j)].norm() - if i == j { 1.0 } else { 0.0 };
            acc += val * val;
        }
    }
    acc.sqrt()
}

/// Resolve the iteration cap. MATLAB's `maxcounter = 50` is DEAD CODE; the reference's
/// real loop bound is `15 * n` (see nonCircComplexFastICAsym.m line 60). `None` therefore
/// resolves to `15 * n`, NOT 50.
pub fn resolve_max_iter(max_iter: Option<usize>, n: usize) -> usize {
    max_iter.unwrap_or(15 * n)
}

/// Noncircular complex FastICA, symmetric orthogonalization.
///
/// `x`: (N, T) complex mixtures. Fully deterministic (the reference has no rand/randn
/// calls). Returns an `EstimatorResult` with `s = w * x`.
///
/// `max_iter`: maximum iterations. If `None` (default), uses the reference's effective
/// cap of `15 * n` (where n is the number of components). The MATLAB reference assigns
/// `maxcounter = 50` but never uses it; the actual loop bound is `15 * n`. Passing an
/// explicit integer overrides this default.
pub fn nc_fastica(
    x: &DMatrix<Complex64>,
    nonlinearity: &str,
    tol: f64,
    max_iter: Option<usize>,
) -> Result<EstimatorResult, String> {
    let nonlin = NonLinearity::parse(nonlinearity)?;

    let (n, m) = x.shape();
    let max_iter = resolve_max_iter(max_iter, n);

    // Whitening: eig(cov(xold')) in MATLAB. MATLAB's cov() computes the CONJUGATE
    // (Hermitian) covariance with N-1 normalization on the MEAN-CENTERED data; the
    // covariance calculation centers internally but the un-centered `x` is what actually
    // gets whitened and returned below (matches np.cov(xold) followed by `Q @ xold`).
    let mut xc = x.clone();
    for i in 0..n {
        let mean: Complex64 =
            (0..m).map(|t| x[(i, t)]).sum::<Complex64>() / Complex64::new(m as f64, 0.0);
        for t in 0..m {
            xc[(i, t)] -= mean;
        }
    }
    let denom = Complex64::new((m as f64) - 1.0, 0.0);
    let cov = (&xc * xc.adjoint()) / denom;

    let eig = cov.symmetric_eigen(); // ascending real eigenvalues, orthonormal (unitary) eigenvectors
    let dx = &eig.eigenvalues;
    let ex = &eig.eigenvectors;

    // Q = sqrt(inv(Dx)) @ Ex'  (Ex' is the CONJUGATE transpose - Ex is unitary)
    let mut q = DMatrix::<Complex64>::zeros(n, n);
    for k in 0..n {
        let s = dx[k].sqrt();
        for i in 0..n {
            q[(k, i)] = ex[(i, k)].conj() / Complex64::new(s, 0.0);
        }
    }

    let xw = &q * x; // whitened, but NOT mean-centered (matches the reference)

    // Pseudo-covariance E[x x^T]: PLAIN transpose (NOT conjugate) - this is what lets the
    // algorithm exploit noncircularity. Share CEBM's named, separately-guarded helper
    // rather than inlining it: a code comment is not a guard. A whole-phase review found
    // that corrupting this to `.adjoint()` inline left ALL 40 tests green (ISI merely
    // doubles, 0.0069 -> 0.0131, well inside the bar), reproducing the exact bug class a
    // Phase-2 review PROVED the ISI oracle cannot catch.
    let pc = pseudo_cov(&xw);

    let mut w = DMatrix::<Complex64>::identity(n, n);
    let mut w_old = DMatrix::<Complex64>::zeros(n, n);
    let mut k = 0usize;

    // Convergence condition: loop while orthonormality deviation exceeds threshold AND
    // iteration count is within the limit (max_iter, which defaults to 15*n).
    while frob_diff(&w_old, &w) > (n as f64) * tol && k < max_iter {
        k += 1;
        w_old = w.clone();

        for kk in 0..n {
            let wold_col: Vec<Complex64> = (0..n).map(|i| w_old[(i, kk)]).collect();

            let mut g_rad = vec![Complex64::new(0.0, 0.0); n];
            let mut ggg_acc = 0.0f64;
            let mut b_acc = Complex64::new(0.0, 0.0);

            for t in 0..m {
                // yy = W(:,kk)' * x
                let mut yy = Complex64::new(0.0, 0.0);
                for i in 0..n {
                    yy += wold_col[i].conj() * xw[(i, t)];
                }
                let absy = yy.norm_sqr();
                let (g, gp) = nonlin.g_gp(absy);

                let coeff = Complex64::new(g, 0.0) * yy.conj();
                for i in 0..n {
                    g_rad[i] += xw[(i, t)] * coeff;
                }
                ggg_acc += gp * absy + g;
                b_acc += Complex64::new(gp, 0.0) * yy.conj() * yy.conj();
            }

            let inv_m = Complex64::new(1.0 / (m as f64), 0.0);
            for gi in g_rad.iter_mut() {
                *gi *= inv_m;
            }
            let ggg = ggg_acc / (m as f64);
            let b_coeff = b_acc * inv_m;

            for i in 0..n {
                let mut s = Complex64::new(0.0, 0.0);
                for j in 0..n {
                    s += pc[(i, j)] * wold_col[j].conj();
                }
                let b_row = b_coeff * s;
                w[(i, kk)] = wold_col[i] * Complex64::new(ggg, 0.0) - g_rad[i] + b_row;
            }
        }

        // Symmetric orthonormalization: W = W * E * inv(sqrt(D)) * E'
        let gram = w.adjoint() * &w;
        let orth = gram.symmetric_eigen();
        let mut inv_sqrt_d = DMatrix::<Complex64>::zeros(n, n);
        for i in 0..n {
            inv_sqrt_d[(i, i)] = Complex64::new(1.0 / orth.eigenvalues[i].sqrt(), 0.0);
        }
        w = &w * &orth.eigenvectors * inv_sqrt_d * orth.eigenvectors.adjoint();
    }

    let q_inv = q
        .try_inverse()
        .ok_or_else(|| "nc_fastica: whitening matrix Q is singular".to_string())?;
    let ahat = q_inv * w;

    let w_out = ahat.try_inverse().ok_or_else(|| {
        "nc_fastica: estimated mixing matrix is singular (pinv failed)".to_string()
    })?;
    let a_out = w_out.clone().try_inverse().ok_or_else(|| {
        "nc_fastica: estimated demixing matrix is singular (pinv failed)".to_string()
    })?;
    let s_out = &w_out * x;

    Ok(EstimatorResult {
        w: w_out,
        a: a_out,
        s: s_out,
    })
}
