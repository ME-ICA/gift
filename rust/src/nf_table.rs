//! Nonlinearity lookup tables for Complex ICA-EBM.
//!
//! The tables are piecewise-polynomial (spline) forms. `simplified_ppval` is an exact port
//! of the evaluator bundled with the reference implementation (complex_ICA_EBM.m:755-802),
//! ported rather than replaced with a library spline so the numerics match the reference.
//!
//! GPL v3 - derives from the Adali-lab (MLSP/UMBC) complex ICA-EBM implementation.
//! Reference: Li & Adali (2010), IEEE Trans. Circuits Syst. I, 57(7):1417-1430.

use std::path::Path;

/// MATLAB piecewise-polynomial form: cubic (order 4), coefficients in descending powers
/// of the local variable (x - breaks[i]).
#[derive(Debug, Clone)]
pub struct Pp {
    pub breaks: Vec<f64>,
    pub coefs: Vec<[f64; 4]>,
    pub pieces: usize,
}

/// One entropy-bound nonlinearity: scalars plus its spline and the spline's slope.
///
/// NOTE: `critical_point2` is present only on nf1; it is NaN for nf2-nf8 by design of the
/// source table. It is inert legacy metadata - the reference algorithm never reads it.
#[derive(Debug, Clone)]
pub struct Nf {
    pub min_egx: f64,
    pub max_egx: f64,
    pub critical_point: f64,
    pub critical_point2: f64,
    pub pp: Pp,
    pub pp_slope: Pp,
}

fn read_npy_f64(path: &Path) -> (Vec<f64>, Vec<usize>, npyz::Order) {
    let bytes =
        std::fs::read(path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let npy = npyz::NpyFile::new(&bytes[..]).expect("valid .npy");
    let shape: Vec<usize> = npy.shape().iter().map(|&d| d as usize).collect();
    let order = npy.order();
    let data = npy.into_vec::<f64>().expect("f64 payload");
    (data, shape, order)
}

fn load_pp(dir: &Path, name: &str, which: &str) -> Pp {
    let (breaks, _, _) = read_npy_f64(&dir.join(format!("{name}_{which}_breaks.npy")));
    let (flat, shape, order) = read_npy_f64(&dir.join(format!("{name}_{which}_coefs.npy")));
    let pieces = shape[0];
    assert_eq!(shape[1], 4, "{name}_{which}: expected order-4 coefficients");
    // These coefficient tables came through scipy.io.loadmat, so unlike most fixtures in this
    // repo they are FORTRAN-ORDER (column-major): element (i, j) of the (pieces, 4) array
    // lives at flat[j*pieces + i], NOT flat[i*4 + j]. Dispatch on the header flag rather than
    // hardcoding either layout - assuming the wrong one silently transposes rows/cols with
    // no error, just wrong spline values downstream.
    let coefs = (0..pieces)
        .map(|i| match order {
            npyz::Order::Fortran => [
                flat[i],
                flat[pieces + i],
                flat[2 * pieces + i],
                flat[3 * pieces + i],
            ],
            npyz::Order::C => [
                flat[i * 4],
                flat[i * 4 + 1],
                flat[i * 4 + 2],
                flat[i * 4 + 3],
            ],
        })
        .collect();
    Pp {
        breaks,
        coefs,
        pieces,
    }
}

pub fn load_nf_table(dir: &Path) -> [Nf; 8] {
    std::array::from_fn(|i| {
        let name = format!("nf{}", i + 1);
        let (s, _, _) = read_npy_f64(&dir.join(format!("{name}_scalars.npy")));
        Nf {
            min_egx: s[0],
            max_egx: s[1],
            critical_point: s[2],
            critical_point2: s[3],
            pp: load_pp(dir, &name, "pp"),
            pp_slope: load_pp(dir, &name, "pp_slope"),
        }
    })
}

/// Exact port of the reference's `simplified_ppval` (complex_ICA_EBM.m:755-802).
///
/// Binary-searches for the piece, shifts to local coordinates, then evaluates by nested
/// (Horner) multiplication. Outside the knot range the index is clamped to the first/last
/// piece and the polynomial extrapolates - the reference does not raise.
pub fn simplified_ppval(pp: &Pp, xs: f64) -> f64 {
    let b = &pp.breaks;
    let c = &pp.coefs;
    let ell = pp.pieces; // MATLAB `l`

    // piece index, in MATLAB's 1-based terms; converted to 0-based below
    let index: usize = if xs > b[ell - 1] {
        // MATLAB: xs > b(l)  =>  index = l
        ell
    } else if xs < b[1] {
        // MATLAB: xs < b(2)  =>  index = 1
        1
    } else {
        let (mut lo, mut hi) = (1usize, ell);
        loop {
            // MATLAB's round() is half-AWAY-FROM-ZERO. Rust's f64::round() is too, but be
            // explicit with floor(x + 0.5) so the intent survives refactoring (Python's
            // built-in round() is banker's rounding, which would be WRONG here).
            let mid = ((lo + hi) as f64 / 2.0 + 0.5).floor() as usize;
            if b[mid - 1] > xs {
                hi = mid;
            } else {
                lo = mid;
            }
            if lo == hi - 1 {
                break lo;
            }
        }
    };

    let i = index - 1; // to 0-based
    let x = xs - b[i];
    let row = &c[i];
    let mut v = row[0];
    // Horner's method over a fixed 4-coefficient cubic: the index form matches the
    // reference's ppval and reads as the polynomial it is.
    #[allow(clippy::needless_range_loop)]
    for j in 1..4 {
        v = x * v + row[j];
    }
    v
}
