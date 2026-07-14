//! Phase-quality mask (quality-map thresholding).
//!
//! Phase is trustworthy only where SNR is high: in brain voxels the complex time series
//! points in a nearly consistent direction over time, while in noise voxels the phase
//! wanders. Temporal phase stability is therefore a proxy for SNR.
//!
//! Reference: Rodriguez, Correa, Eichele, Calhoun & Adali (2011),
//! J. Signal Process. Syst. 65:497-508.

use nalgebra::DMatrix;
use num_complex::Complex64;

/// Q[v] = |sum_t z[v,t]| / sum_t |z[v,t]|, in [0, 1]. `z` is (V, T).
///
/// Invariant to a constant per-voxel phase offset: a unit-modulus factor common to a row
/// leaves both |sum z| and sum |z| unchanged.
pub fn quality_map(z: &DMatrix<Complex64>) -> Vec<f64> {
    z.row_iter()
        .map(|row| {
            let num: Complex64 = row.iter().sum();
            let den: f64 = row.iter().map(|x| x.norm()).sum::<f64>() + f64::EPSILON;
            num.norm() / den
        })
        .collect()
}

/// Otsu threshold over a 256-bin histogram; returns a bin centre.
pub fn otsu_threshold(x: &[f64]) -> f64 {
    const NBINS: usize = 256;
    let finite: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    let lo = finite.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !(hi > lo) {
        return lo;
    }

    let width = (hi - lo) / NBINS as f64;
    let mut counts = [0f64; NBINS];
    for &v in &finite {
        let mut b = ((v - lo) / width).floor() as usize;
        if b >= NBINS {
            b = NBINS - 1; // the maximum lands in the last bin
        }
        counts[b] += 1.0;
    }
    let total: f64 = counts.iter().sum();

    // sigma_b(k) = (muT * omega(k) - mu(k))^2 / (omega(k) * (1 - omega(k)))
    let mut omega = 0.0;
    let mut mu = 0.0;
    let mu_t: f64 = counts
        .iter()
        .enumerate()
        .map(|(i, &c)| (i as f64 + 1.0) * c / total)
        .sum();

    let (mut best_k, mut best) = (0usize, f64::NEG_INFINITY);
    for k in 0..NBINS {
        let p = counts[k] / total;
        omega += p;
        mu += (k as f64 + 1.0) * p;
        let denom = omega * (1.0 - omega);
        if denom <= 0.0 {
            continue;
        }
        let sigma_b = (mu_t * omega - mu).powi(2) / denom;
        if sigma_b > best {
            best = sigma_b;
            best_k = k;
        }
    }

    lo + (best_k as f64 + 0.5) * width // bin centre
}

/// (mask, q, tau). Intersects a magnitude/brain mask with Q > Otsu(Q over the mask).
pub fn phase_quality_mask(
    z: &DMatrix<Complex64>,
    mag_mask: Option<&[bool]>,
) -> (Vec<bool>, Vec<f64>, f64) {
    let v = z.nrows();
    let mag: Vec<bool> = match mag_mask {
        Some(m) => m.to_vec(),
        None => vec![true; v],
    };
    let q = quality_map(z);
    let inside: Vec<f64> = q
        .iter()
        .zip(mag.iter())
        .filter(|(_, &m)| m)
        .map(|(&x, _)| x)
        .collect();
    let tau = otsu_threshold(&inside);
    let mask = q
        .iter()
        .zip(mag.iter())
        .map(|(&x, &m)| m && x > tau)
        .collect();
    (mask, q, tau)
}
