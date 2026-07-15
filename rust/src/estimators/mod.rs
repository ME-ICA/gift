//! Complex ICA estimators. Each returns the same triple.

pub mod cebm;
pub mod nc_fastica;

use nalgebra::DMatrix;
use num_complex::Complex64;

pub struct EstimatorResult {
    pub w: DMatrix<Complex64>, // (N, N) demixing:  s = w * x
    pub a: DMatrix<Complex64>, // (N, N) mixing:    x ~ a * s
    pub s: DMatrix<Complex64>, // (N, T) sources
}
