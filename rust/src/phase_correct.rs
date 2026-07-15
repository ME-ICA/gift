//! Phase-ambiguity correction (the complex-domain analogue of real ICA's sign fix).
//!
//! Complex ICA recovers each source only up to a complex scalar c_k * e^{i theta_k}, so
//! the split of a component's energy between real and imaginary parts is arbitrary.
//! Convention: rotate each component so its energy is maximally concentrated in the real
//! part. The voxel values form an elongated cloud ~ r(v) e^{i alpha}; squaring maps the
//! +/-alpha line ambiguity to the single angle 2*alpha, giving the closed form
//! theta = -0.5 * arg(sum s^2).
//!
//! Reference: Rodriguez, Calhoun & Adali (2012), Pattern Recognition 45:2050-2063.

use nalgebra::DMatrix;
use num_complex::Complex64;

fn skewness(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    x.iter().map(|v| (v - mean).powi(3)).sum::<f64>() / n
}

/// Rotate each component onto the real axis; apply the inverse rotation to `a` so that
/// `a * s` is preserved exactly. Returns the applied `theta` per component.
pub fn correct_phase(
    s: &mut DMatrix<Complex64>,
    a: &mut DMatrix<Complex64>,
    mask: Option<&[bool]>,
) -> Vec<f64> {
    let (n, v) = s.shape();
    let keep: Vec<usize> = match mask {
        Some(m) => (0..v).filter(|&i| m[i]).collect(),
        None => (0..v).collect(),
    };

    let mut theta = vec![0.0; n];
    for k in 0..n {
        // theta = -0.5 * arg(sum of squares over masked voxels)
        let sum_sq: Complex64 = keep.iter().map(|&i| s[(k, i)] * s[(k, i)]).sum();
        let mut th = -0.5 * sum_sq.arg();

        let rot = Complex64::new(th.cos(), th.sin());
        for i in 0..v {
            s[(k, i)] *= rot;
        }

        // residual pi ambiguity: make the real-part skewness positive.
        // Fold the flip into theta BEFORE rotating `a`, or the reconstruction breaks.
        let re: Vec<f64> = keep.iter().map(|&i| s[(k, i)].re).collect();
        if skewness(&re) < 0.0 {
            th += std::f64::consts::PI;
            for i in 0..v {
                s[(k, i)] = -s[(k, i)];
            }
        }

        let inv = Complex64::new(th.cos(), -th.sin()); // e^{-i th}
        for r in 0..a.nrows() {
            a[(r, k)] *= inv;
        }
        theta[k] = th;
    }
    theta
}

/// Align each subject component to a reference (e.g. the group/aggregate map).
///
/// Per-subject phase corrections are independent, so without this the residual rotations
/// reintroduce non-physiological variance before group statistics.
pub fn align_to_reference(
    s: &mut DMatrix<Complex64>,
    s_ref: &DMatrix<Complex64>,
) -> Vec<f64> {
    let (n, v) = s.shape();
    let mut theta = vec![0.0; n];
    for k in 0..n {
        let dot: Complex64 = (0..v).map(|i| s_ref[(k, i)].conj() * s[(k, i)]).sum();
        let th = -dot.arg();
        let rot = Complex64::new(th.cos(), th.sin());
        for i in 0..v {
            s[(k, i)] *= rot;
        }
        theta[k] = th;
    }
    theta
}
