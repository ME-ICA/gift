//! Fixture loading + correctness-oracle helpers shared by the Rust tests.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use nalgebra::DMatrix;
use num_complex::Complex64;

pub fn fixtures_dir() -> PathBuf {
    // rust/tests/common/mod.rs -> repo root is three levels up from CARGO_MANIFEST_DIR/..
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("repo root")
        .join("complex_ica_fixtures")
        .join("npy")
}

fn read_npy<T: npyz::Deserialize>(name: &str) -> (Vec<T>, Vec<usize>, npyz::Order) {
    let path = fixtures_dir().join(format!("{name}.npy"));
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("cannot read fixture {}: {e}", path.display()));
    let npy = npyz::NpyFile::new(&bytes[..]).expect("valid .npy");
    let shape: Vec<usize> = npy.shape().iter().map(|&d| d as usize).collect();
    let order = npy.order();
    let data: Vec<T> = npy.into_vec().expect("npy payload");
    (data, shape, order)
}

/// numpy's `.npy` header carries an explicit `fortran_order` flag, and the payload bytes are
/// laid out however that flag says - NOT always C-order. In particular, arrays that started
/// life as `scipy.io.loadmat` output (cS, A, cX, oracle_*_W here) keep MATLAB's native
/// column-major layout all the way through `np.save`, so their `.npy` files are
/// `fortran_order: True`.
///
/// nalgebra's `DMatrix::from_vec`/`from_column_slice` read column-major, while
/// `from_row_slice` reads row-major. Ignoring the flag and always assuming row-major (or
/// always assuming column-major) SILENTLY TRANSPOSES whichever case you got wrong - no error,
/// just wrong results downstream. So: honour the flag explicitly. A 1-D array becomes a
/// (1, n) row (order is irrelevant for a single row).
fn to_matrix<T: nalgebra::Scalar + Copy + num_traits::Zero>(
    data: Vec<T>,
    shape: Vec<usize>,
    order: npyz::Order,
) -> DMatrix<T> {
    let (r, c) = match shape.len() {
        1 => (1usize, shape[0]),
        2 => (shape[0], shape[1]),
        n => panic!("expected a 1-D or 2-D fixture, got {n} dims"),
    };
    match order {
        npyz::Order::C => DMatrix::from_row_slice(r, c, &data),
        npyz::Order::Fortran => DMatrix::from_column_slice(r, c, &data),
    }
}

pub fn load_c64(name: &str) -> DMatrix<Complex64> {
    let (data, shape, order) = read_npy::<Complex64>(name);
    to_matrix(data, shape, order)
}

pub fn load_f64(name: &str) -> DMatrix<f64> {
    let (data, shape, order) = read_npy::<f64>(name);
    to_matrix(data, shape, order)
}

pub fn load_u8(name: &str) -> Vec<u8> {
    read_npy::<u8>(name).0
}

/// Amari inter-symbol interference of a global matrix G = W * A.
/// 0.0 == perfect separation. Invariant to permutation and to per-source complex
/// scale/phase - which is exactly the indeterminacy of complex ICA.
pub fn isi(g: &DMatrix<Complex64>) -> f64 {
    let n = g.nrows();
    let a = g.map(|z| z.norm()); // |g_ij|
    let mut total = 0.0;
    for i in 0..n {
        let row_max = (0..n).map(|j| a[(i, j)]).fold(0.0, f64::max);
        total += (0..n).map(|j| a[(i, j)] / row_max).sum::<f64>() - 1.0;
    }
    for j in 0..n {
        let col_max = (0..n).map(|i| a[(i, j)]).fold(0.0, f64::max);
        total += (0..n).map(|i| a[(i, j)] / col_max).sum::<f64>() - 1.0;
    }
    total / (2.0 * n as f64 * (n as f64 - 1.0))
}

/// Match each estimated source to a true source by |complex correlation|.
/// Returns (perm, corr): perm[k] is the true-source index best matching estimated source k.
pub fn match_sources(
    s_est: &DMatrix<Complex64>,
    s_true: &DMatrix<Complex64>,
) -> (Vec<usize>, Vec<f64>) {
    let centre_and_normalise = |m: &DMatrix<Complex64>| -> DMatrix<Complex64> {
        let mut out = m.clone();
        for mut row in out.row_iter_mut() {
            let n = row.ncols() as f64;
            let mean: Complex64 = row.iter().sum::<Complex64>() / Complex64::new(n, 0.0);
            row.iter_mut().for_each(|z| *z -= mean);
            let norm = row.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
            row.iter_mut().for_each(|z| *z /= Complex64::new(norm, 0.0));
        }
        out
    };

    let e = centre_and_normalise(s_est);
    let t = centre_and_normalise(s_true);

    let mut perm = Vec::with_capacity(e.nrows());
    let mut corr = Vec::with_capacity(e.nrows());
    for i in 0..e.nrows() {
        let (mut best_j, mut best) = (0usize, -1.0f64);
        for j in 0..t.nrows() {
            // |<e_i, t_j>| : conjugate one side (complex correlation modulus)
            let dot: Complex64 = (0..e.ncols()).map(|k| e[(i, k)] * t[(j, k)].conj()).sum();
            let m = dot.norm();
            if m > best {
                best = m;
                best_j = j;
            }
        }
        perm.push(best_j);
        corr.push(best);
    }
    (perm, corr)
}

/// Reproducible test-data generator (xorshift + Box-Muller). Tests must never depend on a
/// system RNG. Used by Tasks 7, 8 and 10 to build synthetic sources.
pub struct TestRng(u64);

impl TestRng {
    pub fn new(seed: u64) -> Self {
        TestRng(
            seed.wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407)
                | 1,
        )
    }
    /// uniform in [0, 1)
    pub fn uniform(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 11) as f64 / (1u64 << 53) as f64
    }
    /// standard normal
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(f64::MIN_POSITIVE);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}
