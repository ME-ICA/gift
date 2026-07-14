# Complex ICA — Rust Port (Phase 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Rust crate `complex-gift` that runs the complete complex-valued fMRI ICA pipeline (complex NIfTI I/O → phase-quality mask → complex whitening → complex ICA → phase-ambiguity correction → group back-reconstruction), ported module-for-module from the validated Python and held to numeric parity against it.

**Architecture:** The Rust modules mirror the Python modules 1:1 so tests transfer directly. Python (41 tests green, itself oracle-validated against MATLAB) is the **intermediate oracle**: for every unit, Python exports a fixed input and its own output as `.npy`, and the Rust test asserts agreement. Linear algebra is **pure Rust** (`nalgebra`) — there is no system LAPACK on this machine, so `ndarray-linalg` is not an option.

**Tech Stack:** Rust 1.95 / cargo, `nalgebra` 0.34 (complex Hermitian eigendecomposition, SVD, pinv), `num-complex` 0.4 (`Complex64`), `npyz` 0.8 (feature `complex`, reads the `.npy` oracle fixtures bit-exactly), `nifti` 0.17.

## Global Constraints

- **Crate root is `rust/`.** Package name `complex-gift`, library name `complex_gift`. Do not add a `Cargo.toml` at the repo root.
- **Linear algebra: `nalgebra` only.** `ndarray-linalg` requires a system LAPACK backend; this machine has only the `.so.3` runtime libs (no `.so` dev symlinks) and no `gfortran`/`cmake`, so it CANNOT link. Verified working in `nalgebra` 0.34: complex Hermitian eigendecomposition (`.symmetric_eigen()` — real eigenvalues, unitary eigenvectors, reconstruction error 1.5e-14), complex `.svd()`, and `.pseudo_inverse()` (‖MPM−M‖ = 3.9e-15). No general (non-Hermitian) complex eigendecomposition is needed anywhere in this port.
- **License:** GPL v3. `estimators/cebm.rs` and `estimators/nc_fastica.rs` are translations of GPL-v3 Adalí-lab code — each carries a GPL v3 header naming the source `.m` file and the paper.
- **Scalar type:** `num_complex::Complex64` (= `Complex<f64>`) throughout. Matrices are `nalgebra::DMatrix<Complex64>`.
- **Orientation:** `X` is `(N, T)` = rows × cols = components/channels × samples, matching Python and MATLAB. For spatial fMRI ICA, `T` is voxels.
- **Fixtures are READ-ONLY.** `complex_ica_fixtures/` holds the MATLAB oracle. Task 1 adds `complex_ica_fixtures/npy/` (exported once from Python, committed). Never regenerate or hand-edit them.
- **NOT ported: the Strong Uncorrelating Transform.** Phase 2 established that no estimator calls it (both estimators whiten internally; the spec's original claim that nc-FastICA needs the SUT was wrong — see the correction in spec §4). YAGNI: do not port `strong_uncorrelating_transform`. This is also why no general complex eigendecomposition is required.

### THE central hazard: what is and is not elementwise-comparable

Python's `numpy.linalg.eigh` and nalgebra's `symmetric_eigen` return eigenvalues in a **different order** and eigenvectors with a **different arbitrary phase**. Anything downstream of an eigendecomposition therefore does NOT match Python elementwise, even when both are perfectly correct. Comparing it elementwise will produce a failing test that is not a bug — and "fixing" the code to chase it would introduce one.

| Unit | Comparable how |
|---|---|
| `simplified_ppval`, `quality_map`, `otsu_threshold`, `phase_quality_mask` | **Elementwise** (pure arithmetic) — tolerance 1e-10 |
| `correct_phase`, `align_to_reference` | **Elementwise** (pure arithmetic, given fixed inputs) — 1e-10 |
| `back_reconstruct` | **Elementwise** *when fed Python's exact intermediate inputs from the fixture* — 1e-10 |
| `whiten_hermitian`, `two_stage_pca` | **Invariants only**: whitened covariance ≈ I; dewhitening reconstructs; signal subspace matches. NEVER elementwise. |
| `nc_fastica`, `cebm` | **ISI / source correlation only** (permutation- and phase-invariant). NEVER elementwise. CEBM is additionally stochastic. |

---

## Task 1: Crate scaffolding, `.npy` oracle fixtures, and test helpers

**Files:**
- Create: `rust/Cargo.toml`, `rust/src/lib.rs`
- Create: `python/tools/export_rust_fixtures.py`
- Create: `rust/tests/common/mod.rs`
- Create: `rust/tests/helpers.rs`
- Output (git-tracked): `complex_ica_fixtures/npy/*.npy`

**Interfaces:**
- Produces `rust/tests/common/mod.rs`:
  - `fixtures_dir() -> PathBuf` — repo-root `complex_ica_fixtures/npy/`
  - `load_c64(name: &str) -> DMatrix<Complex64>` — loads a complex128 `.npy` into a `(rows, cols)` matrix (numpy is C-order/row-major; nalgebra is column-major — the loader MUST transpose correctly, see Step 4)
  - `load_f64(name: &str) -> DMatrix<f64>`; `load_u8(name: &str) -> Vec<u8>` (masks are exported as u8 0/1, because `.npy` bool support is not worth relying on)
  - `isi(g: &DMatrix<Complex64>) -> f64` — Amari ISI; 0.0 == perfect separation; invariant to permutation and per-source complex scale/phase
  - `match_sources(s_est, s_true) -> (Vec<usize>, Vec<f64>)` — per estimated source, index of best-matching true source and the |complex correlation|
  - `TestRng` — a reproducible PRNG for generating test data (`TestRng::new(seed)`, `.uniform()`, `.normal()`). Tasks 7, 8 and 10 all build synthetic data with it; tests must never depend on a system RNG.

- [ ] **Step 1: Create the crate**

`rust/Cargo.toml`:

```toml
[package]
name = "complex-gift"
version = "0.1.0"
edition = "2021"
description = "Complex-valued fMRI ICA - Rust port of GIFT's complex ICA pipeline"
license = "GPL-3.0-or-later"

[lib]
name = "complex_gift"
path = "src/lib.rs"

[dependencies]
nalgebra = "0.34"
num-complex = "0.4"
nifti = "0.17"
ndarray = "0.16"          # the nifti crate exchanges volumes as ndarray arrays
npyz = { version = "0.8", features = ["complex"] }   # reads the .npy lookup tables

[dev-dependencies]
num-traits = "0.2"
```

`rust/src/lib.rs`:

```rust
//! complex-gift: complex-valued fMRI ICA (Rust port of GIFT's complex ICA pipeline).
//!
//! GPL v3 - derives from the Adali-lab (MLSP/UMBC) complex ICA algorithms.

pub mod nf_table;
```

(Each later task appends its own `pub mod` line. For this task, create `rust/src/nf_table.rs` as an empty placeholder module so the crate compiles: a file containing only `//! Nonlinearity lookup tables (see Task 2).`)

- [ ] **Step 2: Write the fixture exporter**

`python/tools/export_rust_fixtures.py`:

```python
"""Export the oracle fixtures + per-unit parity fixtures as .npy for the Rust port.

Run from `python/`:
    micromamba run -n giftenv python tools/export_rust_fixtures.py

Python is the INTERMEDIATE ORACLE for Rust: for each unit we write a fixed input and
Python's own output, and the Rust test asserts agreement. Python is itself validated
against the MATLAB oracle, so agreement with Python transitively validates against MATLAB.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from complex_gift.nf_table import load_nf_table, simplified_ppval
from complex_gift.phase_correct import align_to_reference, correct_phase
from complex_gift.phase_mask import otsu_threshold, phase_quality_mask, quality_map

FIX = Path(__file__).resolve().parents[2] / "complex_ica_fixtures"
OUT = FIX / "npy"


def save(name, arr):
    np.save(OUT / f"{name}.npy", arr)
    print(f"  {name}.npy {getattr(arr, 'shape', ())} {getattr(arr, 'dtype', type(arr))}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1234)

    # --- ground truth + MATLAB oracle decompositions ---
    d = loadmat(FIX / "complex_sources.mat")
    save("cS", d["cS"]); save("A", d["A"]); save("cX", d["cX"])
    for est in ("cebm", "ncfastica"):
        o = loadmat(FIX / f"oracle_{est}.mat")
        save(f"oracle_{est}_W", o["W"])

    # --- nf_table: scalars + spline coefficients, per nonlinearity ---
    nf = load_nf_table(FIX / "nf_table.mat")
    for name, v in nf.items():
        save(f"{name}_scalars",
             np.array([v.min_EGx, v.max_EGx, v.critical_point, v.critical_point2]))
        for attr in ("pp", "pp_slope"):
            pp = getattr(v, attr)
            save(f"{name}_{attr}_breaks", pp.breaks)
            save(f"{name}_{attr}_coefs", pp.coefs)

    # --- parity: simplified_ppval (pure arithmetic -> elementwise comparable) ---
    xs_all, ys_all = [], []
    for name in [f"nf{i}" for i in range(1, 9)]:
        pp = nf[name].pp
        # sample inside the knot range AND outside it (exercises the clamp branches)
        xs = np.concatenate([
            np.linspace(pp.breaks[0] - 2.0, pp.breaks[-1] + 2.0, 61),
            np.linspace(pp.breaks[0], pp.breaks[-1], 40),
        ])
        xs_all.append(xs)
        ys_all.append(np.array([simplified_ppval(pp, float(x)) for x in xs]))
    save("parity_ppval_xs", np.stack(xs_all))     # (8, 101)
    save("parity_ppval_ys", np.stack(ys_all))     # (8, 101)

    # --- parity: phase mask (pure arithmetic -> elementwise comparable) ---
    V, T = 500, 80
    mag = 1.0 + 0.1 * rng.standard_normal((V, T))
    phi = np.empty((V, T))
    phi[:250] = 0.1 * rng.standard_normal((250, T))       # stable phase
    phi[250:] = 2 * np.pi * rng.random((250, T))          # random phase
    Z = mag * np.exp(1j * phi)
    mask, Q, tau = phase_quality_mask(Z)
    save("parity_mask_Z", Z)
    save("parity_mask_Q", Q)
    save("parity_mask_tau", np.array([tau]))
    save("parity_mask_mask", mask.astype(np.uint8))       # u8, not bool

    # --- parity: phase correction (pure arithmetic -> elementwise comparable) ---
    N, Vv, M = 4, 3000, 20
    S0 = rng.standard_normal((N, Vv)) + 1j * 0.05 * rng.standard_normal((N, Vv))
    A0 = rng.standard_normal((M, N)) + 1j * rng.standard_normal((M, N))
    theta_true = np.array([0.7, -1.1, 0.3, 2.0])
    S_in = S0 * np.exp(1j * theta_true)[:, None]
    A_in = A0 * np.exp(-1j * theta_true)[None, :]
    pc_mask = np.zeros(Vv, dtype=bool); pc_mask[: Vv // 2] = True
    S_out, A_out, theta_out = correct_phase(S_in.copy(), A_in.copy(), mask=pc_mask)
    save("parity_pc_S_in", S_in); save("parity_pc_A_in", A_in)
    save("parity_pc_mask", pc_mask.astype(np.uint8))
    save("parity_pc_S_out", S_out); save("parity_pc_A_out", A_out)
    save("parity_pc_theta", theta_out)

    # --- parity: group alignment ---
    S_ref = rng.standard_normal((3, 1500)) + 1j * 0.05 * rng.standard_normal((3, 1500))
    rot = np.array([0.9, -0.4, 2.2])
    S_subj = S_ref * np.exp(1j * rot)[:, None]
    S_al, th_al = align_to_reference(S_subj.copy(), S_ref)
    save("parity_align_S_ref", S_ref); save("parity_align_S_in", S_subj)
    save("parity_align_S_out", S_al); save("parity_align_theta", th_al)

    print(f"\nwrote fixtures to {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Generate the fixtures**

Run from `python/`:
```bash
/home/tsalo/.local/bin/micromamba run -n giftenv python tools/export_rust_fixtures.py
```
Expected: prints each written file, ending `wrote fixtures to .../complex_ica_fixtures/npy`.

- [ ] **Step 4: Write the failing test**

`rust/tests/helpers.rs`:

```rust
mod common;

use common::{isi, load_c64, match_sources};
use nalgebra::DMatrix;
use num_complex::Complex64;

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

#[test]
fn isi_is_zero_for_a_scaled_permutation() {
    // identity
    let eye = DMatrix::<Complex64>::identity(4, 4);
    assert!(isi(&eye) < 1e-12);

    // a genuinely permuted, arbitrarily phase-scaled global matrix is ALSO perfect
    // separation - that is exactly ICA's indeterminacy, and ISI must be blind to it.
    let perm = [2usize, 0, 3, 1];
    let mut g = DMatrix::<Complex64>::zeros(4, 4);
    for (i, &j) in perm.iter().enumerate() {
        let k = i as f64;
        g[(i, j)] = c((k + 1.0) * (0.3 * k).cos(), (k + 1.0) * (0.3 * k).sin());
    }
    assert!(isi(&g) < 1e-12, "isi = {}", isi(&g));
}

#[test]
fn isi_is_large_for_a_maximally_mixed_matrix() {
    let g = DMatrix::from_element(4, 4, c(1.0, 0.0));
    assert!(isi(&g) > 0.9, "isi = {}", isi(&g));
}

#[test]
fn match_sources_recovers_permutation_through_phase_and_scale() {
    let s = load_c64("cS"); // (6, 4000) ground truth
    let n = s.nrows();
    let perm_true = [2usize, 0, 4, 1, 5, 3];
    let mut est = DMatrix::<Complex64>::zeros(n, s.ncols());
    for (k, &j) in perm_true.iter().enumerate() {
        let scale = c(2.0 * (0.7 * k as f64).cos(), 2.0 * (0.7 * k as f64).sin());
        for t in 0..s.ncols() {
            est[(k, t)] = s[(j, t)] * scale;
        }
    }
    let (perm, corr) = match_sources(&est, &s);
    assert_eq!(perm, perm_true.to_vec());
    assert!(corr.iter().all(|&x| x > 0.99), "corr = {:?}", corr);
}

#[test]
fn fixtures_load_and_satisfy_the_ground_truth_identity() {
    let cs = load_c64("cS");
    let a = load_c64("A");
    let cx = load_c64("cX");
    assert_eq!(cs.shape(), (6, 4000));
    assert_eq!(a.shape(), (6, 6));
    // the whole oracle rests on cX == A @ cS
    let recon = &a * &cs;
    let err = (&recon - &cx).norm() / cx.norm();
    assert!(err < 1e-12, "cX != A @ cS (rel err {err:.3e})");
}
```

- [ ] **Step 5: Run to verify it fails**

Run: `cd rust && cargo test --test helpers 2>&1 | tail -20`
Expected: FAIL — `file not found for module 'common'` / unresolved imports.

- [ ] **Step 6: Implement the test helpers**

`rust/tests/common/mod.rs`:

```rust
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

/// numpy's `.npy` header carries an explicit `fortran_order` flag, and `into_vec()` returns
/// the bytes IN THE ORDER THEY ARE STORED - it does NOT normalise to C-order.
///
/// Do not assume C-order! Arrays that came through `scipy.io.loadmat` keep MATLAB's
/// column-major layout, so `np.save` writes them with `fortran_order: True` (this covers
/// cS/A/cX, the oracle W matrices, AND every nf*_coefs table). Arrays constructed fresh in
/// numpy are C-order.
///
/// nalgebra's `from_column_slice` reads column-major and `from_row_slice` reads row-major.
/// Guessing wrong SILENTLY TRANSPOSES the matrix - no error, just wrong numbers everywhere
/// downstream. Dispatch on the flag. A 1-D array becomes a (1, n) row (order is irrelevant).
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
            let dot: Complex64 = (0..e.ncols())
                .map(|k| e[(i, k)] * t[(j, k)].conj())
                .sum();
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
        TestRng(seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) | 1)
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
```

Add `num-traits = "0.2"` to `[dev-dependencies]` in `rust/Cargo.toml` (used by `to_matrix`).

- [ ] **Step 7: Run to verify it passes**

Run: `cd rust && cargo test --test helpers 2>&1 | tail -8`
Expected: `test result: ok. 4 passed`.

- [ ] **Step 8: Commit**

```bash
git add rust/ python/tools/export_rust_fixtures.py complex_ica_fixtures/npy/
git commit -m "feat(rust): crate scaffolding, .npy oracle fixtures, ISI/match_sources helpers"
```

---

## Task 2: `nf_table` — lookup tables and `simplified_ppval`

**Files:**
- Create: `rust/src/nf_table.rs` (replacing the placeholder)
- Modify: `rust/src/lib.rs` (module already declared in Task 1)
- Create: `rust/tests/nf_table.rs`

**Interfaces:**
- Consumes: `complex_ica_fixtures/npy/nf{1..8}_{scalars,pp_breaks,pp_coefs,pp_slope_breaks,pp_slope_coefs}.npy` (Task 1).
- Produces:
  - `pub struct Pp { pub breaks: Vec<f64>, pub coefs: Vec<[f64; 4]>, pub pieces: usize }`
  - `pub struct Nf { pub min_egx: f64, pub max_egx: f64, pub critical_point: f64, pub critical_point2: f64, pub pp: Pp, pub pp_slope: Pp }`
  - `pub fn load_nf_table(dir: &Path) -> [Nf; 8]` — index 0 is `nf1`.
  - `pub fn simplified_ppval(pp: &Pp, xs: f64) -> f64`

**Background:** `simplified_ppval` is an exact port of the evaluator bundled with the reference MATLAB (`complex_ICA_EBM.m:755-802`), already ported to Python and verified against `scipy.interpolate.PPoly`. Port the Python, which is the verified version. Two details that MUST be preserved: the binary search uses `floor((low+high)/2 + 0.5)` (MATLAB rounds half away from zero, Python's `round` is banker's — in Rust use `((lo + hi) as f64 / 2.0 + 0.5).floor() as usize`), and the order is hard-coded to 4 (the reference does this itself). `critical_point2` is NaN for nf2–nf8 by design — inert legacy metadata, not a bug.

- [ ] **Step 1: Write the failing parity test**

`rust/tests/nf_table.rs`:

```rust
mod common;

use common::{fixtures_dir, load_f64};
use complex_gift::nf_table::{load_nf_table, simplified_ppval};

#[test]
fn tables_have_the_expected_structure() {
    let nf = load_nf_table(&fixtures_dir());
    for (i, v) in nf.iter().enumerate() {
        assert_eq!(v.pp.breaks.len(), v.pp.pieces + 1, "nf{}", i + 1);
        assert_eq!(v.pp.coefs.len(), v.pp.pieces, "nf{}", i + 1);
        assert!(v.max_egx > v.min_egx, "nf{}", i + 1);
    }
    // only nf1 carries critical_point2; nf2-nf8 are NaN by design (inert legacy metadata)
    assert!(nf[0].critical_point2.is_finite());
    assert!(nf[1].critical_point2.is_nan());
}

#[test]
fn ppval_matches_python_elementwise() {
    // Pure arithmetic => this IS elementwise-comparable against the Python oracle.
    // The sampled xs deliberately run outside the knot range, exercising the clamp
    // branches of the binary search as well as the interior.
    let nf = load_nf_table(&fixtures_dir());
    let xs = load_f64("parity_ppval_xs"); // (8, 101)
    let ys = load_f64("parity_ppval_ys"); // (8, 101)

    for k in 0..8 {
        for j in 0..xs.ncols() {
            let got = simplified_ppval(&nf[k].pp, xs[(k, j)]);
            let want = ys[(k, j)];
            let tol = 1e-10 * want.abs().max(1.0);
            assert!(
                (got - want).abs() < tol,
                "nf{} at x={}: rust {} vs python {}",
                k + 1,
                xs[(k, j)],
                got,
                want
            );
        }
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd rust && cargo test --test nf_table 2>&1 | tail -10`
Expected: FAIL — unresolved imports (`load_nf_table`, `simplified_ppval` do not exist).

- [ ] **Step 3: Implement**

`rust/src/nf_table.rs`:

```rust
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

fn read_npy_f64(path: &Path) -> (Vec<f64>, Vec<usize>) {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let npy = npyz::NpyFile::new(&bytes[..]).expect("valid .npy");
    let shape: Vec<usize> = npy.shape().iter().map(|&d| d as usize).collect();
    (npy.into_vec::<f64>().expect("f64 payload"), shape)
}

fn load_pp(dir: &Path, name: &str, which: &str) -> Pp {
    let (breaks, _) = read_npy_f64(&dir.join(format!("{name}_{which}_breaks.npy")));
    let (flat, shape) = read_npy_f64(&dir.join(format!("{name}_{which}_coefs.npy")));
    let pieces = shape[0];
    assert_eq!(shape[1], 4, "{name}_{which}: expected order-4 coefficients");
    // numpy is row-major: row i is coefs[i*4 .. i*4+4]
    let coefs = (0..pieces)
        .map(|i| [flat[i * 4], flat[i * 4 + 1], flat[i * 4 + 2], flat[i * 4 + 3]])
        .collect();
    Pp { breaks, coefs, pieces }
}

pub fn load_nf_table(dir: &Path) -> [Nf; 8] {
    std::array::from_fn(|i| {
        let name = format!("nf{}", i + 1);
        let (s, _) = read_npy_f64(&dir.join(format!("{name}_scalars.npy")));
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
    for j in 1..4 {
        v = x * v + row[j];
    }
    v
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd rust && cargo test --test nf_table 2>&1 | tail -8`
Expected: `test result: ok. 2 passed`.

- [ ] **Step 5: Commit**

```bash
git add rust/
git commit -m "feat(rust): nf_table loader + exact simplified_ppval port (parity with Python)"
```

---

## Task 3: Complex Hermitian whitening

**Files:**
- Create: `rust/src/whiten.rs`
- Modify: `rust/src/lib.rs` (add `pub mod whiten;`)
- Create: `rust/tests/whiten.rs`

**Interfaces:**
- Produces:
  - `pub struct Whitening { pub xw: DMatrix<Complex64>, pub w_whiten: DMatrix<Complex64>, pub w_dewhiten: DMatrix<Complex64> }`
  - `pub fn whiten_hermitian(x: &DMatrix<Complex64>, n_components: usize) -> Result<Whitening, String>`
    `x` is `(P, T)`; `xw` is `(N, T)`; `w_whiten` is `(N, P)`; `w_dewhiten` is `(P, N)`.

**Mirrors the Python `whiten_hermitian`, including its two guards** (Phase 2 added these after a review found silent failures): return `Err` if `n_components > P`, and `Err` if the smallest retained eigenvalue is not comfortably positive (`<= 1e-12 * largest`) — a rank-deficient input must never yield a silently NaN-contaminated result.

> **Do NOT compare `xw` or `w_whiten` elementwise against Python.** nalgebra's `symmetric_eigen` and numpy's `eigh` order eigenvalues differently and produce eigenvectors with a different arbitrary phase, so the whitened rows differ by permutation and phase even when both are exactly right. Test the invariants instead.

- [ ] **Step 1: Write the failing test**

`rust/tests/whiten.rs`:

```rust
mod common;

use common::load_c64;
use complex_gift::whiten::whiten_hermitian;
use nalgebra::DMatrix;
use num_complex::Complex64;

#[test]
fn whitening_makes_the_covariance_identity() {
    let x = load_c64("cX"); // (6, 4000)
    let n = 6;
    let w = whiten_hermitian(&x, n).expect("whiten");
    assert_eq!(w.xw.shape(), (n, x.ncols()));

    let t = x.ncols() as f64;
    let cov = (&w.xw * w.xw.adjoint()) / Complex64::new(t, 0.0);
    let err = (&cov - DMatrix::<Complex64>::identity(n, n)).norm();
    assert!(err < 1e-8, "whitened covariance is not identity (err {err:.3e})");
}

#[test]
fn dewhitening_reconstructs_the_centred_data() {
    let x = load_c64("cX");
    let n = 6; // data is exactly rank 6, so dewhitening must reconstruct it
    let w = whiten_hermitian(&x, n).expect("whiten");

    // centre x along its own axis-1 (the sample axis), as whiten_hermitian does
    let t = x.ncols();
    let mut xc = x.clone();
    for mut row in xc.row_iter_mut() {
        let mean: Complex64 = row.iter().sum::<Complex64>() / Complex64::new(t as f64, 0.0);
        row.iter_mut().for_each(|z| *z -= mean);
    }

    let recon = &w.w_dewhiten * &w.xw;
    let err = (&recon - &xc).norm() / xc.norm();
    assert!(err < 1e-10, "dewhitening did not reconstruct (rel err {err:.3e})");
}

#[test]
fn errors_when_more_components_than_rows() {
    let x = load_c64("cX"); // 6 rows
    assert!(whiten_hermitian(&x, 12).is_err());
}

#[test]
fn errors_on_rank_deficient_input() {
    // 6 rows spanning only 3 independent directions -> asking for 6 must fail loudly
    // rather than returning NaNs from sqrt of a ~0 eigenvalue.
    let base = load_c64("cS").rows(0, 3).into_owned(); // (3, 4000)
    let mut x = DMatrix::<Complex64>::zeros(6, base.ncols());
    for i in 0..6 {
        let src = i % 3;
        let scale = Complex64::new(1.0 + i as f64, 0.5 * i as f64);
        for j in 0..base.ncols() {
            x[(i, j)] = base[(src, j)] * scale;
        }
    }
    assert!(whiten_hermitian(&x, 6).is_err());
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd rust && cargo test --test whiten 2>&1 | tail -10`
Expected: FAIL — unresolved import `complex_gift::whiten`.

- [ ] **Step 3: Implement**

`rust/src/whiten.rs`:

```rust
//! Complex whitening using the Hermitian covariance E[x x^H].
//!
//! Mirrors the Python `whiten_hermitian` (which mirrors GIFT's icatb_pca_whitening.m).
//!
//! The Strong Uncorrelating Transform is deliberately NOT ported: no estimator calls it
//! (both estimators whiten internally), so porting it would be dead code.

use nalgebra::DMatrix;
use num_complex::Complex64;

pub struct Whitening {
    pub xw: DMatrix<Complex64>,        // (N, T)
    pub w_whiten: DMatrix<Complex64>,  // (N, P)
    pub w_dewhiten: DMatrix<Complex64>, // (P, N)
}

pub fn whiten_hermitian(
    x: &DMatrix<Complex64>,
    n_components: usize,
) -> Result<Whitening, String> {
    let (p, t) = x.shape();
    if n_components > p {
        return Err(format!(
            "whiten_hermitian: n_components={n_components} exceeds the {p} available rows"
        ));
    }
    let n = n_components;

    // centre along the sample axis
    let mut xc = x.clone();
    for mut row in xc.row_iter_mut() {
        let mean: Complex64 = row.iter().sum::<Complex64>() / Complex64::new(t as f64, 0.0);
        row.iter_mut().for_each(|z| *z -= mean);
    }

    // Hermitian covariance (conjugate transpose - NOT the plain transpose)
    let r = (&xc * xc.adjoint()) / Complex64::new(t as f64, 0.0);

    let eig = r.clone().symmetric_eigen();
    // sort eigenvalues descending, keep the top N
    let mut order: Vec<usize> = (0..p).collect();
    order.sort_by(|&a, &b| {
        eig.eigenvalues[b]
            .partial_cmp(&eig.eigenvalues[a])
            .expect("eigenvalues are real and finite")
    });
    let keep = &order[..n];

    let d_max = eig.eigenvalues[order[0]];
    let d_min = eig.eigenvalues[*keep.last().expect("n >= 1")];
    if !(d_min > 1e-12 * d_max) {
        return Err(format!(
            "whiten_hermitian: data is rank-deficient at n_components={n} (smallest \
             retained eigenvalue {d_min:.3e} vs largest {d_max:.3e}); refusing to return \
             a silently NaN-contaminated result"
        ));
    }

    let mut w_whiten = DMatrix::<Complex64>::zeros(n, p);
    let mut w_dewhiten = DMatrix::<Complex64>::zeros(p, n);
    for (k, &idx) in keep.iter().enumerate() {
        let s = eig.eigenvalues[idx].sqrt();
        for i in 0..p {
            let u = eig.eigenvectors[(i, idx)];
            w_whiten[(k, i)] = u.conj() / Complex64::new(s, 0.0);
            w_dewhiten[(i, k)] = u * Complex64::new(s, 0.0);
        }
    }

    let xw = &w_whiten * &xc;
    Ok(Whitening { xw, w_whiten, w_dewhiten })
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd rust && cargo test --test whiten 2>&1 | tail -8`
Expected: `test result: ok. 4 passed`.

- [ ] **Step 5: Commit**

```bash
git add rust/
git commit -m "feat(rust): complex Hermitian whitening with rank guards"
```

---

## Task 4: Phase-quality mask

**Files:**
- Create: `rust/src/phase_mask.rs`
- Modify: `rust/src/lib.rs` (add `pub mod phase_mask;`)
- Create: `rust/tests/phase_mask.rs`

**Interfaces:**
- Produces:
  - `pub fn quality_map(z: &DMatrix<Complex64>) -> Vec<f64>` — `z` is `(V, T)`; `Q[v] = |sum_t z[v,t]| / sum_t |z[v,t]|` in `[0,1]`.
  - `pub fn otsu_threshold(x: &[f64]) -> f64` — 256-bin histogram Otsu.
  - `pub fn phase_quality_mask(z: &DMatrix<Complex64>, mag_mask: Option<&[bool]>) -> (Vec<bool>, Vec<f64>, f64)` — returns `(mask, q, tau)`.

**The defining property:** `Q` is invariant to a constant per-voxel phase offset (both `|sum z|` and `sum |z|` are unchanged), which is what lets the mask run on raw complex data before any background-phase removal.

> **Otsu parity caveat:** Python uses `skimage.filters.threshold_otsu` (256-bin histogram over `[min, max]`, returning a bin centre). The Rust port implements the same 256-bin method, so `tau` should agree closely — but binning details can differ. The test asserts `Q` elementwise (exact arithmetic, tight) and `tau` to 1e-6; if `tau` proves to differ by more than that, do NOT loosen the tolerance — investigate the histogram construction, since a divergent `tau` means the two ports select different voxels.

- [ ] **Step 1: Write the failing parity test**

`rust/tests/phase_mask.rs`:

```rust
mod common;

use common::{load_c64, load_f64, load_u8};
use complex_gift::phase_mask::{otsu_threshold, phase_quality_mask, quality_map};

#[test]
fn quality_map_matches_python_elementwise() {
    // Pure arithmetic => elementwise-comparable against the Python oracle.
    let z = load_c64("parity_mask_Z");       // (500, 80)
    let q_py = load_f64("parity_mask_Q");    // (1, 500)
    let q = quality_map(&z);
    assert_eq!(q.len(), q_py.ncols());
    for v in 0..q.len() {
        assert!(
            (q[v] - q_py[(0, v)]).abs() < 1e-12,
            "voxel {v}: rust {} vs python {}",
            q[v],
            q_py[(0, v)]
        );
    }
}

#[test]
fn mask_and_threshold_match_python() {
    let z = load_c64("parity_mask_Z");
    let tau_py = load_f64("parity_mask_tau")[(0, 0)];
    let mask_py = load_u8("parity_mask_mask");

    let (mask, _q, tau) = phase_quality_mask(&z, None);
    assert!(
        (tau - tau_py).abs() < 1e-6,
        "otsu threshold diverged: rust {tau} vs python {tau_py}"
    );
    let disagreements = mask
        .iter()
        .zip(mask_py.iter())
        .filter(|(&r, &p)| r != (p == 1))
        .count();
    assert_eq!(disagreements, 0, "mask disagrees with Python on {disagreements} voxels");
}

#[test]
fn quality_map_is_invariant_to_a_constant_per_voxel_phase() {
    // THE defining property: a static per-voxel phase offset (B0/receiver phase) must not
    // change Q at all. This is what lets the mask run before background-phase removal.
    let z = load_c64("parity_mask_Z");
    let q1 = quality_map(&z);

    let mut z2 = z.clone();
    for (v, mut row) in z2.row_iter_mut().enumerate() {
        // one constant phase per voxel, applied to that voxel's whole time series
        let phi = 0.7 * v as f64;
        let rot = num_complex::Complex64::new(phi.cos(), phi.sin());
        row.iter_mut().for_each(|x| *x *= rot);
    }
    let q2 = quality_map(&z2);
    for v in 0..q1.len() {
        assert!((q1[v] - q2[v]).abs() < 1e-12, "Q changed at voxel {v}");
    }
}

#[test]
fn otsu_separates_two_well_separated_modes() {
    let mut x = Vec::new();
    for i in 0..500 {
        x.push(0.10 + 0.001 * (i % 20) as f64);
    }
    for i in 0..500 {
        x.push(0.90 + 0.001 * (i % 20) as f64);
    }
    let tau = otsu_threshold(&x);
    // Otsu's objective is flat across the empty gap, so argmax may land anywhere in it -
    // assert the functional property (it separates the modes), not a tie-break location.
    assert!(x[..500].iter().filter(|&&v| v < tau).count() >= 495);
    assert!(x[500..].iter().filter(|&&v| v > tau).count() >= 495);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd rust && cargo test --test phase_mask 2>&1 | tail -10`
Expected: FAIL — unresolved import `complex_gift::phase_mask`.

- [ ] **Step 3: Implement**

`rust/src/phase_mask.rs`:

```rust
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd rust && cargo test --test phase_mask 2>&1 | tail -8`
Expected: `test result: ok. 4 passed`.

- [ ] **Step 5: Commit**

```bash
git add rust/
git commit -m "feat(rust): phase-quality mask (quality map + Otsu), parity with Python"
```

---

## Task 5: Phase-ambiguity correction and group alignment

**Files:**
- Create: `rust/src/phase_correct.rs`
- Modify: `rust/src/lib.rs` (add `pub mod phase_correct;`)
- Create: `rust/tests/phase_correct.rs`

**Interfaces:**
- Produces:
  - `pub fn correct_phase(s: &mut DMatrix<Complex64>, a: &mut DMatrix<Complex64>, mask: Option<&[bool]>) -> Vec<f64>` — rotates in place, returns `theta`. `s` is `(N, V)`, `a` is `(M, N)`.
  - `pub fn align_to_reference(s: &mut DMatrix<Complex64>, s_ref: &DMatrix<Complex64>) -> Vec<f64>` — rotates in place, returns `theta`.

**The maths (mirrors the Python):** per component `k`, `theta[k] = -0.5 * arg(sum_v s[k,v]^2)` over masked voxels, then `s[k,:] *= e^{i theta}` and `a[:,k] *= e^{-i theta}` — which cancel, so `a * s` is preserved EXACTLY. Then a residual π sign fix: if the skewness of the real part (over masked voxels) is negative, add π (equivalently negate the component) — and the sign flip must be folded into `theta` BEFORE rotating `a`, or the reconstruction silently breaks.

For alignment: `theta[k] = -arg(sum_v conj(s_ref[k,v]) * s[k,v])`, then `s[k,:] *= e^{i theta}`.

- [ ] **Step 1: Write the failing parity test**

`rust/tests/phase_correct.rs`:

```rust
mod common;

use common::{load_c64, load_f64, load_u8};
use complex_gift::phase_correct::{align_to_reference, correct_phase};

#[test]
fn correct_phase_matches_python_elementwise() {
    // Pure arithmetic given fixed inputs => elementwise-comparable against Python.
    let mut s = load_c64("parity_pc_S_in");
    let mut a = load_c64("parity_pc_A_in");
    let mask: Vec<bool> = load_u8("parity_pc_mask").iter().map(|&b| b == 1).collect();

    let s_py = load_c64("parity_pc_S_out");
    let a_py = load_c64("parity_pc_A_out");
    let theta_py = load_f64("parity_pc_theta");

    let theta = correct_phase(&mut s, &mut a, Some(&mask));

    for k in 0..theta.len() {
        assert!(
            (theta[k] - theta_py[(0, k)]).abs() < 1e-10,
            "theta[{k}]: rust {} vs python {}",
            theta[k],
            theta_py[(0, k)]
        );
    }
    assert!((&s - &s_py).norm() / s_py.norm() < 1e-10, "S diverged from Python");
    assert!((&a - &a_py).norm() / a_py.norm() < 1e-10, "A diverged from Python");
}

#[test]
fn correct_phase_preserves_the_reconstruction() {
    // The whole point: rotating S by e^{i0} and A by e^{-i0} must leave A*S untouched -
    // including when the residual pi sign-flip branch fires.
    let mut s = load_c64("parity_pc_S_in");
    let mut a = load_c64("parity_pc_A_in");
    let before = &a * &s;

    let _ = correct_phase(&mut s, &mut a, None);

    let after = &a * &s;
    let err = (&before - &after).norm() / before.norm();
    assert!(err < 1e-10, "A*S was not preserved (rel err {err:.3e})");
}

#[test]
fn align_to_reference_matches_python_and_collapses_known_rotations() {
    let s_ref = load_c64("parity_align_S_ref");
    let mut s = load_c64("parity_align_S_in");
    let s_py = load_c64("parity_align_S_out");
    let theta_py = load_f64("parity_align_theta");

    let theta = align_to_reference(&mut s, &s_ref);

    for k in 0..theta.len() {
        assert!(
            (theta[k] - theta_py[(0, k)]).abs() < 1e-10,
            "theta[{k}]: rust {} vs python {}",
            theta[k],
            theta_py[(0, k)]
        );
    }
    assert!((&s - &s_py).norm() / s_py.norm() < 1e-10, "aligned S diverged from Python");
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd rust && cargo test --test phase_correct 2>&1 | tail -10`
Expected: FAIL — unresolved import `complex_gift::phase_correct`.

- [ ] **Step 3: Implement**

`rust/src/phase_correct.rs`:

```rust
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd rust && cargo test --test phase_correct 2>&1 | tail -8`
Expected: `test result: ok. 3 passed`.

- [ ] **Step 5: Commit**

```bash
git add rust/
git commit -m "feat(rust): phase-ambiguity correction + group alignment (parity with Python)"
```

---

## Task 6: Noncircular complex FastICA

**Files:**
- Create: `rust/src/estimators/mod.rs`, `rust/src/estimators/nc_fastica.rs`
- Modify: `rust/src/lib.rs` (add `pub mod estimators;`)
- Create: `rust/tests/nc_fastica.rs`

**Sources to translate (read BOTH completely before writing Rust):**
- `python/complex_gift/estimators/nc_fastica.py` (the validated port — translate THIS)
- `GroupICAT/icatb/icatb_analysis_functions/icatb_algorithms/complex_ica/nonCircComplexFastICAsym.m` (the original, for reference)

**Interfaces:**
- Produces:
  - `pub struct EstimatorResult { pub w: DMatrix<Complex64>, pub a: DMatrix<Complex64>, pub s: DMatrix<Complex64> }` (in `estimators/mod.rs`)
  - `pub fn nc_fastica(x: &DMatrix<Complex64>, nonlinearity: &str, tol: f64, max_iter: Option<usize>) -> Result<EstimatorResult, String>`
    `nonlinearity` in `{"log", "kurt", "sqrt"}`. `max_iter: None` resolves to `15 * n`.

**Facts established in Phase 2 (rely on them):**
- nc-FastICA is **fully deterministic** — the reference has ZERO `rand`/`randn` calls. No seeding.
- MATLAB's `tol = 1e-5` and `maxcounter = 50` locals are **dead code**. The real loop condition is `while (norm(abs(Wold'*W)-eye(n),'fro') > (n*1e-5) && k < 15*n)`. So the effective threshold is `n*tol` and the effective cap is **`15*n`, not 50**.
- **Transpose hazard:** the covariance uses the **conjugate** transpose (`.adjoint()`); the **pseudo-covariance uses the PLAIN transpose** (`.transpose()`). Confusing them destroys the noncircularity the algorithm exists to exploit while still appearing to run. In nalgebra: `x.adjoint()` is conjugate-transpose, `x.transpose()` is plain.
- Its whitening uses an eigendecomposition, so the recovered components may come out in a **different permutation and phase** than Python's. Compare via ISI and |correlation| — **never elementwise**.

- [ ] **Step 1: Write the failing test**

`rust/tests/nc_fastica.rs`:

```rust
mod common;

use common::{isi, load_c64, match_sources};
use complex_gift::estimators::nc_fastica::nc_fastica;

#[test]
fn separates_the_shared_fixture_as_well_as_the_matlab_reference() {
    // nc-FastICA is deterministic, but its internal eigendecomposition means the component
    // ORDER and PHASE may differ from Python/MATLAB. Compare with ISI (permutation- and
    // phase-invariant) against the KNOWN true mixing - never elementwise.
    let cx = load_c64("cX");
    let a_true = load_c64("A");
    let cs = load_c64("cS");
    let w_matlab = load_c64("oracle_ncfastica_W");

    let isi_matlab = isi(&(&w_matlab * &a_true));

    let res = nc_fastica(&cx, "log", 1e-5, None).expect("nc_fastica");
    assert_eq!(res.w.shape(), (6, 6));
    assert_eq!(res.s.shape(), (6, cx.ncols()));

    let isi_rust = isi(&(&res.w * &a_true));
    assert!(isi_rust < 0.05, "port separates poorly: ISI={isi_rust:.5}");
    assert!(
        isi_rust < 2.0 * isi_matlab + 0.01,
        "port is materially worse than MATLAB: rust={isi_rust:.5} matlab={isi_matlab:.5}"
    );

    let (perm, corr) = match_sources(&res.s, &cs);
    let mut seen = perm.clone();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), 6, "not a bijective permutation: {perm:?}");
    assert!(corr.iter().all(|&c| c > 0.9), "weak recovery: {corr:?}");
}

#[test]
fn default_max_iter_is_the_reference_cap_of_15n() {
    // MATLAB's `maxcounter = 50` is DEAD CODE; the real cap is 15*n. Guard the default so
    // nobody "restores" 50 and silently stops up to 3x early for realistic model orders.
    let cx = load_c64("cX"); // n = 6 -> 15*n = 90 != 50
    let a = nc_fastica(&cx, "log", 1e-5, None).expect("default");
    let b = nc_fastica(&cx, "log", 1e-5, Some(15 * cx.nrows())).expect("explicit 15n");
    assert!((&a.w - &b.w).norm() < 1e-12, "default cap is not 15*n");
}

#[test]
fn rejects_an_unknown_nonlinearity() {
    let cx = load_c64("cX");
    assert!(nc_fastica(&cx, "bogus", 1e-5, None).is_err());
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd rust && cargo test --test nc_fastica 2>&1 | tail -10`
Expected: FAIL — unresolved import `complex_gift::estimators`.

- [ ] **Step 3: Create the estimator module**

`rust/src/estimators/mod.rs`:

```rust
//! Complex ICA estimators. Each returns the same triple.

pub mod nc_fastica;

use nalgebra::DMatrix;
use num_complex::Complex64;

pub struct EstimatorResult {
    pub w: DMatrix<Complex64>, // (N, N) demixing:  s = w * x
    pub a: DMatrix<Complex64>, // (N, N) mixing:    x ~ a * s
    pub s: DMatrix<Complex64>, // (N, T) sources
}
```

- [ ] **Step 4: Port the estimator**

Create `rust/src/estimators/nc_fastica.rs` as a faithful translation of the validated Python
`python/complex_gift/estimators/nc_fastica.py` (cross-checking against the MATLAB original).
Requirements:
- GPL v3 header naming `nonCircComplexFastICAsym.m` and citing Novey & Adali (2008), *On
  extending the complex FastICA algorithm to noncircular sources*, IEEE TSP 56(5):2148-2154.
- Signature exactly `nc_fastica(x: &DMatrix<Complex64>, nonlinearity: &str, tol: f64, max_iter: Option<usize>) -> Result<EstimatorResult, String>`; `None` resolves to `15 * n`; an unknown `nonlinearity` returns `Err`.
- Keep the reference's structure: whiten via the eigendecomposition of the covariance; form the pseudo-covariance with the **plain** transpose; run the fixed-point updates for the three nonlinearities; symmetric orthogonalisation; loop while `‖ |Wold^H W| − I ‖_F > n*tol && k < max_iter`.
- Return `w` such that `s = w * x`, with `a = pinv(w)` and `s = w * x`.
- `a2 = 0.05` is a live constant in the reference (unlike `tol`/`maxcounter`) — keep it.

*(This step translates an existing, already-validated implementation rather than writing new
logic, so the plan pins the contract, the constants, and the hazards; the test in Step 1 is
the executable specification of correctness.)*

- [ ] **Step 5: Run to verify it passes**

Run: `cd rust && cargo test --test nc_fastica 2>&1 | tail -8`
Expected: `test result: ok. 3 passed`.

- [ ] **Step 6: Commit**

```bash
git add rust/
git commit -m "feat(rust): noncircular complex FastICA (oracle-validated)"
```

---

## Task 7: Complex ICA-EBM — the heaviest task

**Files:**
- Create: `rust/src/estimators/cebm.rs`
- Modify: `rust/src/estimators/mod.rs` (add `pub mod cebm;`)
- Create: `rust/tests/cebm.rs`

**Sources to translate (read BOTH completely before writing Rust):**
- `python/complex_gift/estimators/cebm.py` (the validated port — translate THIS)
- `GroupICAT/icatb/icatb_analysis_functions/icatb_algorithms/complex_ica/complex_ICA_EBM.m` (the 802-line original, for reference)

**Interfaces:**
- Consumes: `complex_gift::nf_table::{load_nf_table, simplified_ppval}` (Task 2) — **import them, do not reimplement**.
- Produces: `pub fn cebm(x: &DMatrix<Complex64>, nf_dir: &Path, seed: u64, tol: f64, max_iter: Option<usize>) -> Result<EstimatorResult, String>`

**Stochasticity — this drives the whole test design:**
CEBM is **not** deterministic (random init plus stochastic search inside the optimiser). It therefore CANNOT reproduce Python's or MATLAB's `w` elementwise, and no test may try. Instead:
- Take an explicit `seed: u64` and drive **every** random draw from one seeded generator, so a Rust run is reproducible. (Add a tiny, self-contained PRNG — see below — rather than pulling in `rand`, keeping the dependency surface minimal. It does NOT need to match Python's stream; it only needs to be reproducible.)
- Validate by **separation quality** (ISI against the known true mixing), benchmarked against the MATLAB reference's ISI on the same fixture — exactly as the Python port is validated.

**Required PRNG** (put it in `cebm.rs`; a standard xorshift + Box–Muller):

```rust
/// Minimal reproducible PRNG. CEBM is stochastic; a Rust run must be repeatable given a
/// seed. It does NOT need to match Python's random stream - only to be deterministic.
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) | 1)
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
```

**Translation hazards:**
- **Conjugate vs plain transpose.** The covariance in `pre_processing` uses `.adjoint()`; the **pseudo-covariance uses `.transpose()`**. In Phase 2 a reviewer PROVED the ISI test cannot detect a swap here (deliberately broken variants still passed the ISI bar) — so the Python port carries a dedicated `_pseudo_cov` guard test. **This port must carry the equivalent guard** (Step 1 includes it).
- The `N > 7` branch uses an incremental Sherman–Morrison `inv_Q` update instead of a direct inverse. It is the path real fMRI runs take, and the fixture (N=6) does NOT exercise it — Step 1 includes an N=10 test.
- Port `pre_processing` and `inv_sqrtm_h` using nalgebra's `symmetric_eigen` (Hermitian).

- [ ] **Step 1: Write the failing test**

`rust/tests/cebm.rs`:

```rust
mod common;

use common::{fixtures_dir, isi, load_c64, match_sources};
use complex_gift::estimators::cebm::{cebm, pseudo_cov};
use nalgebra::DMatrix;
use num_complex::Complex64;

#[test]
fn is_reproducible_for_a_fixed_seed() {
    let cx = load_c64("cX");
    let a = cebm(&cx, &fixtures_dir(), 0, 1e-4, None).expect("cebm");
    let b = cebm(&cx, &fixtures_dir(), 0, 1e-4, None).expect("cebm");
    assert!((&a.w - &b.w).norm() < 1e-12, "same seed must give the same W");
}

#[test]
fn pseudo_cov_uses_the_plain_transpose() {
    // Guard the single most likely silent bug. A Phase-2 review PROVED that swapping this
    // to the conjugate transpose still passes the ISI bar - so ISI alone cannot catch it.
    let x = load_c64("cS").rows(0, 3).into_owned();
    let t = x.ncols() as f64;

    let got = pseudo_cov(&x);
    let want_plain = (&x * x.transpose()) / Complex64::new(t, 0.0);
    let want_conj = (&x * x.adjoint()) / Complex64::new(t, 0.0);

    assert!((&got - &want_plain).norm() < 1e-9, "pseudo-cov must use the PLAIN transpose");
    assert!(
        (&got - &want_conj).norm() > 1e-3,
        "pseudo-cov must NOT be the Hermitian covariance"
    );
}

#[test]
fn separates_the_shared_fixture_as_well_as_the_matlab_reference() {
    // CEBM is stochastic, so MATLAB's W is NOT reproducible elementwise. The meaningful
    // question is whether the port separates the fixture as well as the reference does,
    // measured against the KNOWN true mixing A.
    let cx = load_c64("cX");
    let a_true = load_c64("A");
    let cs = load_c64("cS");
    let w_matlab = load_c64("oracle_cebm_W");

    let isi_matlab = isi(&(&w_matlab * &a_true));

    let res = cebm(&cx, &fixtures_dir(), 0, 1e-4, None).expect("cebm");
    let isi_rust = isi(&(&res.w * &a_true));

    assert!(isi_rust < 0.05, "port separates poorly: ISI={isi_rust:.5}");
    assert!(
        isi_rust < 2.0 * isi_matlab + 0.01,
        "port is materially worse than MATLAB: rust={isi_rust:.5} matlab={isi_matlab:.5}"
    );

    let (perm, corr) = match_sources(&res.s, &cs);
    let mut seen = perm.clone();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), 6, "not a bijective permutation: {perm:?}");
    assert!(corr.iter().all(|&c| c > 0.9), "weak recovery: {corr:?}");
}

#[test]
fn separates_at_n_10_exercising_the_incremental_branch() {
    // N > 7 takes the incremental Sherman-Morrison inv_Q path - the one real fMRI runs
    // use, and the one the N=6 fixture never reaches.
    let n = 10usize;
    let t = 5000usize;
    let mut rng = common::TestRng::new(7);

    // super-Gaussian sources: ICA cannot separate Gaussian sources at all
    let mut s = DMatrix::<Complex64>::zeros(n, t);
    for i in 0..n {
        for j in 0..t {
            let re = rng.normal() * rng.normal().abs().powf(1.5);
            let im = rng.normal() * rng.normal().abs().powf(1.5);
            s[(i, j)] = Complex64::new(re, im);
        }
    }
    let a_mix = DMatrix::<Complex64>::from_fn(n, n, |_, _| {
        Complex64::new(rng.normal(), rng.normal())
    });
    let x = &a_mix * &s;

    let res = cebm(&x, &fixtures_dir(), 0, 1e-4, None).expect("cebm at n=10");
    let isi_rust = isi(&(&res.w * &a_mix));
    assert!(isi_rust < 0.05, "n=10 (incremental branch) ISI={isi_rust:.5}");
}
```

(`common::TestRng` already exists from Task 1 — do not redefine it.)

- [ ] **Step 2: Run to verify it fails**

Run: `cd rust && cargo test --test cebm 2>&1 | tail -10`
Expected: FAIL — unresolved import `complex_gift::estimators::cebm`.

- [ ] **Step 3: Port the helpers**

In `rust/src/estimators/cebm.rs`, start with the small, exactly-specifiable pieces:

```rust
use nalgebra::DMatrix;
use num_complex::Complex64;

/// Pseudo-covariance E[x x^T] - PLAIN transpose, NOT the conjugate transpose.
///
/// This is the quantity that carries the noncircularity CEBM exploits. Using `.adjoint()`
/// here silently destroys it while still appearing to work - and the ISI oracle test
/// CANNOT detect the swap (proven in Phase 2), which is why `pseudo_cov` is a named,
/// separately-guarded function rather than an inline expression.
pub fn pseudo_cov(x: &DMatrix<Complex64>) -> DMatrix<Complex64> {
    let t = x.ncols() as f64;
    (x * x.transpose()) / Complex64::new(t, 0.0)
}

/// Inverse matrix square root of a Hermitian matrix (MATLAB `inv_sqrtmH`).
pub(crate) fn inv_sqrtm_h(b: &DMatrix<Complex64>) -> DMatrix<Complex64> {
    let eig = b.clone().symmetric_eigen();
    let n = b.nrows();
    let mut out = DMatrix::<Complex64>::zeros(n, n);
    for k in 0..n {
        let s = Complex64::new(1.0 / eig.eigenvalues[k].sqrt(), 0.0);
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
pub(crate) fn pre_processing(
    x: &DMatrix<Complex64>,
) -> (DMatrix<Complex64>, DMatrix<Complex64>) {
    let t = x.ncols();
    let mut xc = x.clone();
    for mut row in xc.row_iter_mut() {
        let mean: Complex64 = row.iter().sum::<Complex64>() / Complex64::new(t as f64, 0.0);
        row.iter_mut().for_each(|z| *z -= mean);
    }
    // conjugate transpose here: this is the COVARIANCE, not the pseudo-covariance
    let r = (&xc * xc.adjoint()) / Complex64::new(t as f64, 0.0);
    let p = inv_sqrtm_h(&r);
    let xw = &p * &xc;
    (xw, p)
}
```

- [ ] **Step 4: Port the optimiser**

Translate the rest of `python/complex_gift/estimators/cebm.py` (the SEA initial-guess loop and
the `north` optimiser) into `cebm.rs`. Requirements:
- GPL v3 header naming `complex_ICA_EBM.m` and citing Li & Adali (2010), *Complex
  independent component analysis by entropy bound minimization*, IEEE Trans. Circuits
  Syst. I, 57(7):1417-1430.
- Import `simplified_ppval` and `load_nf_table` from `crate::nf_table`; **do not
  reimplement them**.
- Every random draw comes from the seeded `Rng` defined above — no global/system RNG.
- Use `pseudo_cov` (Step 3) at BOTH pseudo-covariance sites; never write `x * x.transpose()`
  inline again.
- Return `EstimatorResult { w, a: pinv(w), s: w * x }`, with `w` mapping the ORIGINAL `x` to
  sources (the pre-processing whitener `p` folded in, exactly as the Python does).
- The Python is the authority — it is validated against MATLAB. Where the Python and the
  MATLAB appear to differ, follow the Python and note it in your report.

*(This step translates an existing, validated ~500-line numerical implementation rather than
writing new logic; the plan pins the contract, structure, stochasticity handling and hazards,
and the tests in Step 1 are the executable specification of correctness.)*

- [ ] **Step 5: Run to verify it passes**

Run: `cd rust && cargo test --test cebm 2>&1 | tail -8`
Expected: `test result: ok. 4 passed`.

- [ ] **Step 6: Sanity-check the transpose guard actually guards**

Temporarily change `pseudo_cov` to use `.adjoint()`, run `cargo test --test cebm`, and
confirm `pseudo_cov_uses_the_plain_transpose` FAILS (and note whether the ISI tests still
pass — in Phase 2 they did, which is the whole reason this guard exists). Then revert.
Report what you observed. A guard that cannot fail is worthless.

- [ ] **Step 7: Commit**

```bash
git add rust/
git commit -m "feat(rust): Complex ICA-EBM port (oracle-validated) + transpose guard"
```

---

## Task 8: Group two-stage PCA and GICA back-reconstruction

**Files:**
- Create: `rust/src/group.rs`
- Modify: `rust/src/lib.rs` (add `pub mod group;`)
- Create: `rust/tests/group.rs`

**Interfaces:**
- Consumes: `complex_gift::whiten::whiten_hermitian` (Task 3).
- Produces:
  - `pub struct Reduction { pub xg: DMatrix<Complex64>, pub reduced: Vec<DMatrix<Complex64>>, pub whiteners: Vec<DMatrix<Complex64>>, pub w_group: DMatrix<Complex64> }`
  - `pub fn two_stage_pca(subjects: &[DMatrix<Complex64>], n_subject: usize, n_group: usize) -> Result<Reduction, String>` — each subject is `(T_i, V)`.
  - `pub fn back_reconstruct(a_group: &DMatrix<Complex64>, red: &Reduction) -> Vec<(DMatrix<Complex64>, DMatrix<Complex64>)>` — returns `(S_i (N,V), A_i (T_i,N))` per subject.

**Two things that are easy to get wrong (both cost a cycle in Phase 2):**
1. `two_stage_pca` must remove the **per-voxel TEMPORAL mean** (`subject[(t, v)] -= mean over t`) before whitening. That strips the static complex baseline image, which is constant over time and would otherwise form a huge rank-1 nuisance direction that consumes a PCA slot and silently discards a real source. This is a DIFFERENT operation from the mean `whiten_hermitian` removes internally (which is along its own sample axis). Both are needed. It must run AFTER the phase mask, never before.
2. `back_reconstruct` must produce SUBJECT-SPECIFIC maps: `B = pinv(W_group) * A_group`, partitioned by subject into blocks `Bi` `(n_subject, N)`, then `S_i = pinv(Bi) * Y_i` using each subject's **actual reduced data** `Y_i`. Writing `S_i = pinv(Bi) * (Bi * S_group)` instead collapses algebraically to `S_group` for EVERY subject (since `pinv(Bi) * Bi = I`) — a silent no-op carrying zero subject-specific variance. `A_i = pinv(W_i) * Bi`.

- [ ] **Step 1: Write the failing test**

`rust/tests/group.rs`:

```rust
mod common;

use common::{match_sources, TestRng};
use complex_gift::group::{back_reconstruct, two_stage_pca};
use complex_gift::whiten::whiten_hermitian;
use nalgebra::DMatrix;
use num_complex::Complex64;

/// Build `n_sub` subjects that genuinely DIFFER: shared maps plus a per-subject
/// perturbation, plus subject-specific noise (which also lifts the rank above N so a
/// reduction can legitimately keep n_subject > n_components).
fn make_subjects(
    rng: &mut TestRng,
    n: usize,
    v: usize,
    t: usize,
    n_sub: usize,
) -> (Vec<DMatrix<Complex64>>, Vec<DMatrix<Complex64>>) {
    let smaps = DMatrix::<f64>::from_fn(n, v, |_, _| {
        rng.normal() * rng.normal().abs().powf(1.5)
    });
    let phi: Vec<f64> = (0..v).map(|_| (rng.uniform() * 20.0 - 10.0) / 180.0 * std::f64::consts::PI).collect();

    let mut subjects = Vec::new();
    let mut truths = Vec::new();
    for _ in 0..n_sub {
        let si = DMatrix::<f64>::from_fn(n, v, |i, j| {
            smaps[(i, j)] + 0.5 * rng.normal() * rng.normal().abs().powf(1.5)
        });
        let tc = DMatrix::<f64>::from_fn(t, n, |_, _| rng.normal());

        // z(v,t) = rho(v,t) * e^{i phi(v)}: POSITIVE magnitude (static baseline + BOLD-like
        // modulation + subject noise) times a STATIC per-voxel background phase.
        let modulation = &tc * &si; // (t, v)
        let mut z = DMatrix::<Complex64>::zeros(t, v);
        for tt in 0..t {
            for vv in 0..v {
                let mag = 100.0 + 5.0 * modulation[(tt, vv)] + 2.0 * rng.normal();
                let rot = Complex64::new(phi[vv].cos(), phi[vv].sin());
                z[(tt, vv)] = Complex64::new(mag, 0.0) * rot;
            }
        }
        subjects.push(z);

        let mut strue = DMatrix::<Complex64>::zeros(n, v);
        for i in 0..n {
            for j in 0..v {
                let rot = Complex64::new(phi[j].cos(), phi[j].sin());
                strue[(i, j)] = Complex64::new(si[(i, j)], 0.0) * rot;
            }
        }
        truths.push(strue);
    }
    (subjects, truths)
}

#[test]
fn two_stage_pca_reduces_to_the_requested_order() {
    let mut rng = TestRng::new(3);
    let (subjects, _) = make_subjects(&mut rng, 3, 400, 60, 4);
    let red = two_stage_pca(&subjects, 6, 3).expect("reduction");
    assert_eq!(red.xg.nrows(), 3);
    assert_eq!(red.xg.ncols(), 400);
    assert_eq!(red.reduced.len(), 4);
    assert_eq!(red.reduced[0].shape(), (6, 400));
}

#[test]
fn back_reconstruction_is_subject_specific() {
    // Guard against the silent no-op: if back_reconstruct collapsed to S_group, every
    // subject would get identical maps and this assertion would fail.
    let mut rng = TestRng::new(4);
    let (n, v, t, n_sub) = (3usize, 400usize, 60usize, 4usize);
    let (subjects, truths) = make_subjects(&mut rng, n, v, t, n_sub);

    let red = two_stage_pca(&subjects, 6, n).expect("reduction");

    // stand in for ICA with a plain whitening of the group data: back-reconstruction is a
    // linear-algebra property and does not depend on WHICH unmixing we use.
    let w = whiten_hermitian(&red.xg, n).expect("whiten");
    let s_group = w.xw.clone();
    let a_group = w.w_dewhiten.clone();

    let subs = back_reconstruct(&a_group, &red);
    assert_eq!(subs.len(), n_sub);

    for (i, (s_i, a_i)) in subs.iter().enumerate() {
        assert_eq!(s_i.shape(), (n, v));
        assert_eq!(a_i.nrows(), t);

        let (_, c_subject) = match_sources(s_i, &truths[i]);
        let (_, c_group) = match_sources(&s_group, &truths[i]);
        let mean_subject: f64 = c_subject.iter().sum::<f64>() / n as f64;
        let mean_group: f64 = c_group.iter().sum::<f64>() / n as f64;
        assert!(
            mean_subject > mean_group,
            "subject {i}: back-reconstruction ({mean_subject:.4}) must explain the subject \
             better than the group maps ({mean_group:.4}) - a collapse to S_group would \
             make these equal"
        );
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd rust && cargo test --test group 2>&1 | tail -10`
Expected: FAIL — unresolved import `complex_gift::group`.

- [ ] **Step 3: Implement**

`rust/src/group.rs`:

```rust
//! Group complex ICA: two-stage PCA reduction and GICA back-reconstruction.
//!
//! Follows GIFT's standard scheme: reduce each subject, temporally concatenate, reduce
//! again at the group level, run ICA on the group-reduced data, then back-reconstruct
//! per-subject maps and time courses.

use nalgebra::DMatrix;
use num_complex::Complex64;

use crate::whiten::whiten_hermitian;

pub struct Reduction {
    pub xg: DMatrix<Complex64>,                 // (n_group, V)
    pub reduced: Vec<DMatrix<Complex64>>,       // per subject: (n_subject, V)
    pub whiteners: Vec<DMatrix<Complex64>>,     // per subject: (n_subject, T_i)
    pub w_group: DMatrix<Complex64>,            // (n_group, n_sub * n_subject)
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
            let mean: Complex64 =
                col.iter().sum::<Complex64>() / Complex64::new(t as f64, 0.0);
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
    Ok(Reduction { xg: g.xw, reduced, whiteners, w_group: g.w_whiten })
}

/// GICA back-reconstruction to subject-specific maps and time courses.
///
/// The group model is Xg = A_group * S_group in the group-reduced space. Undo the group
/// whitener to express the group mixing in the STACKED subject-reduced space:
///
///     B = pinv(W_group) * A_group        // (n_sub * n_subject, N)
///
/// and partition B by subject into Bi (n_subject, N). Bi maps the sources into subject i's
/// reduced space, so subject i's own maps come from projecting that subject's ACTUAL
/// reduced data through it:
///
///     S_i = pinv(Bi) * Y_i               // (N, V)
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

        let pinv_bi = bi.clone().pseudo_inverse(1e-12).expect("pinv of the subject block");
        let s_i = pinv_bi * y_i; // (N, V) - uses the subject's ACTUAL reduced data

        let pinv_wi = w_i.clone().pseudo_inverse(1e-12).expect("pinv of the subject whitener");
        let a_i = pinv_wi * &bi; // (T_i, N)

        out.push((s_i, a_i));
    }
    out
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd rust && cargo test --test group 2>&1 | tail -8`
Expected: `test result: ok. 2 passed`.

- [ ] **Step 5: Sanity-check the subject-specific guard actually guards**

Temporarily replace `let s_i = pinv_bi * y_i;` with `let s_i = pinv_bi * (&bi * &s_group);`
(you will need the group sources in scope — a quick local hack is fine), confirm
`back_reconstruction_is_subject_specific` FAILS, then revert. Report what you observed.

- [ ] **Step 6: Commit**

```bash
git add rust/
git commit -m "feat(rust): group two-stage PCA + GICA back-reconstruction"
```

---

## Task 9: Complex NIfTI I/O

**Files:**
- Create: `rust/src/complex_io.rs`
- Modify: `rust/src/lib.rs` (add `pub mod complex_io;`)
- Create: `rust/tests/complex_io.rs`

**Interfaces:**
- Produces:
  - `pub enum ComplexType { RealImag, MagPhase }`
  - `pub fn read_complex(first: &Path, second: &Path, kind: ComplexType) -> Result<(Vec<Complex64>, [usize; 4]), String>` — returns the flattened volume and its dims `[x, y, z, t]` (a 3-D file yields `t = 1`). Mirrors GIFT's `icatb_loadData.m:79-85`: RealImag → `first + i*second`; MagPhase → `first*cos(second) + i*first*sin(second)`.
  - `pub fn write_complex(data: &[Complex64], dims: [usize; 3], first: &Path, second: &Path, kind: ComplexType) -> Result<(), String>` — splits a complex volume back into two NIfTI files (RealImag → re/im; MagPhase → |z|/arg(z)).
  - `pub fn complex_file_pair(path: &Path, naming: (&str, &str)) -> Result<(PathBuf, PathBuf), String>` — GIFT's convention is a prefix around an underscore; errors (message containing `underscore`) if the filename has none.

- [ ] **Step 1: Write the failing test**

`rust/tests/complex_io.rs`:

```rust
mod common;

use std::path::Path;

use complex_gift::complex_io::{complex_file_pair, read_complex, ComplexType};

// The fixtures directory holds two small NIfTI volumes written by nibabel for this test:
// complex_ica_fixtures/npy/../nifti/{R_probe.nii, I_probe.nii} (see Step 3).
fn nifti_dir() -> std::path::PathBuf {
    common::fixtures_dir().parent().unwrap().join("nifti")
}

#[test]
fn reads_a_real_imaginary_pair() {
    let d = nifti_dir();
    let (data, dims) =
        read_complex(&d.join("R_probe.nii"), &d.join("I_probe.nii"), ComplexType::RealImag)
            .expect("read");
    assert_eq!(dims[0] * dims[1] * dims[2] * dims[3], data.len());
    // the probe volume was written as re = index, im = -index
    assert!((data[5].re - 5.0).abs() < 1e-9);
    assert!((data[5].im + 5.0).abs() < 1e-9);
}

#[test]
fn file_pair_naming_follows_the_gift_convention() {
    let (a, b) = complex_file_pair(Path::new("/data/sub01_run1.nii"), ("R_", "I_")).unwrap();
    assert_eq!(a.file_name().unwrap(), "R_sub01_run1.nii");
    assert_eq!(b.file_name().unwrap(), "I_sub01_run1.nii");
}

#[test]
fn file_pair_requires_an_underscore() {
    let err = complex_file_pair(Path::new("/data/sub01.nii"), ("R_", "I_")).unwrap_err();
    assert!(err.contains("underscore"), "unhelpful error: {err}");
}

#[test]
fn write_then_read_roundtrips() {
    use complex_gift::complex_io::write_complex;
    use num_complex::Complex64;

    let dir = std::env::temp_dir().join("cg_rust_io_roundtrip");
    std::fs::create_dir_all(&dir).unwrap();
    let dims = [4usize, 4, 2];
    let n = dims[0] * dims[1] * dims[2];
    let data: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new(i as f64 * 0.5, -(i as f64) * 0.25))
        .collect();

    let (f1, f2) = (dir.join("R_out.nii"), dir.join("I_out.nii"));
    write_complex(&data, dims, &f1, &f2, ComplexType::RealImag).expect("write");

    let (back, d) = read_complex(&f1, &f2, ComplexType::RealImag).expect("read");
    assert_eq!([d[0], d[1], d[2]], dims);
    for i in 0..n {
        assert!((back[i] - data[i]).norm() < 1e-9, "roundtrip lost value at {i}");
    }
}
```

- [ ] **Step 2: Generate the NIfTI test volumes**

Append to `python/tools/export_rust_fixtures.py` (inside `main()`, before the final print):

```python
    # --- small NIfTI probe volumes for the Rust complex_io tests ---
    import nibabel as nib
    nii = FIX / "nifti"
    nii.mkdir(parents=True, exist_ok=True)
    dims = (4, 4, 2)
    idx = np.arange(int(np.prod(dims)), dtype=np.float64).reshape(dims)
    nib.save(nib.Nifti1Image(idx, np.eye(4)), str(nii / "R_probe.nii"))
    nib.save(nib.Nifti1Image(-idx, np.eye(4)), str(nii / "I_probe.nii"))
    print(f"  nifti/R_probe.nii, nifti/I_probe.nii {dims}")
```

Run from `python/`:
```bash
/home/tsalo/.local/bin/micromamba run -n giftenv python tools/export_rust_fixtures.py
```

- [ ] **Step 3: Run to verify the test fails**

Run: `cd rust && cargo test --test complex_io 2>&1 | tail -10`
Expected: FAIL — unresolved import `complex_gift::complex_io`.

- [ ] **Step 4: Implement**

`rust/src/complex_io.rs`:

```rust
//! Complex NIfTI I/O.
//!
//! GIFT stores complex data as TWO ordinary NIfTI files per volume (image-domain
//! reconstructed data, not raw k-space), distinguished by a prefix around an underscore:
//! R_/I_ for real&imaginary, Mag_/Phase_ for magnitude&phase.
//! Mirrors icatb_loadData.m:79-85.

use std::path::{Path, PathBuf};

use nifti::{NiftiObject, NiftiVolume, ReaderOptions};
use num_complex::Complex64;

#[derive(Debug, Clone, Copy)]
pub enum ComplexType {
    RealImag,
    MagPhase,
}

fn read_volume(path: &Path) -> Result<(Vec<f64>, [usize; 4]), String> {
    let obj = ReaderOptions::new()
        .read_file(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let vol = obj.volume();
    let d = vol.dim();
    let dims = [
        *d.first().unwrap_or(&1) as usize,
        *d.get(1).unwrap_or(&1) as usize,
        *d.get(2).unwrap_or(&1) as usize,
        *d.get(3).unwrap_or(&1) as usize,
    ];
    let data: Vec<f64> = vol
        .into_ndarray::<f64>()
        .map_err(|e| format!("cannot decode {}: {e}", path.display()))?
        .into_raw_vec_and_offset()
        .0;
    Ok((data, dims))
}

/// Assemble one complex volume from GIFT's two-file representation.
pub fn read_complex(
    first: &Path,
    second: &Path,
    kind: ComplexType,
) -> Result<(Vec<Complex64>, [usize; 4]), String> {
    let (a, dims_a) = read_volume(first)?;
    let (b, dims_b) = read_volume(second)?;
    if dims_a != dims_b {
        return Err(format!("shape mismatch: {dims_a:?} vs {dims_b:?}"));
    }
    let data = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| match kind {
            ComplexType::RealImag => Complex64::new(x, y),
            ComplexType::MagPhase => Complex64::new(x * y.cos(), x * y.sin()),
        })
        .collect();
    Ok((data, dims_a))
}

/// Split a complex volume back into GIFT's two-file representation.
pub fn write_complex(
    data: &[Complex64],
    dims: [usize; 3],
    first: &Path,
    second: &Path,
    kind: ComplexType,
) -> Result<(), String> {
    use nifti::writer::WriterOptions;

    let n = dims[0] * dims[1] * dims[2];
    if data.len() != n {
        return Err(format!("data has {} values, dims imply {n}", data.len()));
    }
    let (a, b): (Vec<f64>, Vec<f64>) = data
        .iter()
        .map(|z| match kind {
            ComplexType::RealImag => (z.re, z.im),
            ComplexType::MagPhase => (z.norm(), z.arg()),
        })
        .unzip();

    for (path, vals) in [(first, a), (second, b)] {
        let arr = ndarray::Array3::from_shape_vec((dims[0], dims[1], dims[2]), vals)
            .map_err(|e| format!("bad shape for {}: {e}", path.display()))?;
        WriterOptions::new(path)
            .write_nifti(&arr)
            .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    }
    Ok(())
}

/// Derive GIFT's two filenames from a base name (a prefix around an underscore).
pub fn complex_file_pair(
    path: &Path,
    naming: (&str, &str),
) -> Result<(PathBuf, PathBuf), String> {
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| format!("not a file path: {}", path.display()))?;
    if !name.contains('_') {
        return Err(format!(
            "complex file names must contain an underscore (GIFT convention): {name}"
        ));
    }
    let dir = path.parent().unwrap_or(Path::new("."));
    Ok((
        dir.join(format!("{}{}", naming.0, name)),
        dir.join(format!("{}{}", naming.1, name)),
    ))
}
```

Add `ndarray = "0.16"` to `[dependencies]` in `rust/Cargo.toml` — the `nifti` crate's reader
and writer exchange data as `ndarray` arrays. (This is `ndarray` alone; do NOT add
`ndarray-linalg`, which cannot link here.)

- [ ] **Step 5: Run to verify it passes**

Run: `cd rust && cargo test --test complex_io 2>&1 | tail -8`
Expected: `test result: ok. 4 passed`.

- [ ] **Step 6: Commit**

```bash
git add rust/ python/tools/export_rust_fixtures.py complex_ica_fixtures/nifti/
git commit -m "feat(rust): complex NIfTI I/O (R_/I_ and Mag_/Phase_)"
```

---

## Task 10: Pipeline and end-to-end

**Files:**
- Create: `rust/src/pipeline.rs`
- Modify: `rust/src/lib.rs` (add `pub mod pipeline;`)
- Create: `rust/tests/pipeline.rs`

**Interfaces:**
- Consumes every unit from Tasks 2–8.
- Produces:
  - `pub enum Estimator { Cebm { seed: u64 }, NcFastica }`
  - `pub struct PipelineResult { pub s_group: DMatrix<Complex64>, pub a_group: DMatrix<Complex64>, pub mask: Vec<bool>, pub subjects: Vec<(DMatrix<Complex64>, DMatrix<Complex64>)> }`
  - `pub fn run_complex_ica(subjects: &[DMatrix<Complex64>], n_components: usize, n_subject: Option<usize>, estimator: Estimator, nf_dir: &Path) -> Result<PipelineResult, String>` — each subject is `(T_i, V)`.

**Order of operations (mirrors the Python; the ordering matters):**
1. Phase-quality mask on the **RAW** concatenated complex data `(V, T_total)` — before any de-meaning, because the mask keys on the phase stability that the static baseline provides.
2. Apply the mask to each subject's voxels.
3. `two_stage_pca` (which removes the per-voxel temporal mean internally).
4. The estimator.
5. `correct_phase` on the group maps.
6. `back_reconstruct`, then `align_to_reference` per subject — **and counter-rotate `A_i` by the returned theta**, or the returned pair stops satisfying `X_i ≈ A_i * S_i`.

- [ ] **Step 1: Write the failing end-to-end test**

`rust/tests/pipeline.rs`:

```rust
mod common;

use common::{fixtures_dir, match_sources, TestRng};
use complex_gift::pipeline::{run_complex_ica, Estimator};
use nalgebra::DMatrix;
use num_complex::Complex64;

#[test]
fn end_to_end_group_complex_ica() {
    let mut rng = TestRng::new(21);
    let (n, v_sig, v_noise, t, n_sub) = (3usize, 1000usize, 300usize, 60usize, 4usize);
    let v = v_sig + v_noise;

    // super-Gaussian spatial maps (ICA cannot separate Gaussian sources at all)
    let smaps = DMatrix::<f64>::from_fn(n, v_sig, |_, _| {
        rng.normal() * rng.normal().abs().powf(1.5)
    });
    let scale = smaps.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
    let smaps = smaps / scale;

    // static per-voxel background phase, small spread (+/-10 deg)
    let phi: Vec<f64> = (0..v_sig)
        .map(|_| (rng.uniform() * 20.0 - 10.0) / 180.0 * std::f64::consts::PI)
        .collect();

    let mut subjects = Vec::new();
    for _ in 0..n_sub {
        let tc = DMatrix::<f64>::from_fn(t, n, |_, _| rng.normal());
        let modulation = &tc * &smaps; // (t, v_sig)
        let mut z = DMatrix::<Complex64>::zeros(t, v);
        for tt in 0..t {
            // signal voxels: POSITIVE magnitude x STATIC per-voxel phase -> phase-stable
            for vv in 0..v_sig {
                let mag = 100.0 + 10.0 * rng.uniform() + 5.0 * modulation[(tt, vv)];
                assert!(mag > 0.0);
                z[(tt, vv)] = Complex64::new(mag, 0.0)
                    * Complex64::new(phi[vv].cos(), phi[vv].sin());
            }
            // noise voxels: random phase per timepoint -> the mask must drop them
            for vv in v_sig..v {
                let mag = 0.5 + rng.uniform();
                let ang = std::f64::consts::TAU * rng.uniform();
                z[(tt, vv)] = Complex64::new(mag * ang.cos(), mag * ang.sin());
            }
        }
        subjects.push(z);
    }

    let res = run_complex_ica(
        &subjects,
        n,
        None,
        Estimator::Cebm { seed: 0 },
        &fixtures_dir(),
    )
    .expect("pipeline");

    // the phase mask kept the signal voxels and dropped the random-phase ones
    let kept_sig = res.mask[..v_sig].iter().filter(|&&m| m).count() as f64 / v_sig as f64;
    let kept_noise = res.mask[v_sig..].iter().filter(|&&m| m).count() as f64 / v_noise as f64;
    assert!(kept_sig > 0.9, "mask dropped signal voxels: kept {kept_sig:.3}");
    assert!(kept_noise < 0.05, "mask kept noise voxels: kept {kept_noise:.3}");

    // the group decomposition recovered the true maps (up to permutation/phase).
    // Build the truth over ALL voxels (noise voxels carry no signal) and apply the SAME
    // mask, so a leaked noise voxel cannot cause a shape mismatch.
    let mut strue_full = DMatrix::<Complex64>::zeros(n, v);
    for i in 0..n {
        for j in 0..v_sig {
            strue_full[(i, j)] = Complex64::new(smaps[(i, j)], 0.0)
                * Complex64::new(phi[j].cos(), phi[j].sin());
        }
    }
    let kept: Vec<usize> = (0..v).filter(|&j| res.mask[j]).collect();
    let mut strue = DMatrix::<Complex64>::zeros(n, kept.len());
    for (c, &j) in kept.iter().enumerate() {
        for i in 0..n {
            strue[(i, c)] = strue_full[(i, j)];
        }
    }

    let (perm, corr) = match_sources(&res.s_group, &strue);
    let mut seen = perm.clone();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), n, "not a bijective permutation: {perm:?}");
    assert!(corr.iter().all(|&c| c > 0.8), "weak recovery: {corr:?}");

    // phase correction left the maps concentrated on the real axis
    for i in 0..n {
        let (mut im, mut tot) = (0.0f64, 0.0f64);
        for j in 0..res.s_group.ncols() {
            im += res.s_group[(i, j)].im.powi(2);
            tot += res.s_group[(i, j)].norm_sqr();
        }
        assert!(im / tot < 0.15, "component {i} is not real-axis aligned");
    }

    assert_eq!(res.subjects.len(), n_sub);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd rust && cargo test --test pipeline 2>&1 | tail -10`
Expected: FAIL — unresolved import `complex_gift::pipeline`.

- [ ] **Step 3: Implement**

`rust/src/pipeline.rs`:

```rust
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd rust && cargo test --test pipeline 2>&1 | tail -8`
Expected: `test result: ok. 1 passed`.

- [ ] **Step 5: Run the whole suite**

Run: `cd rust && cargo test 2>&1 | tail -15`
Expected: every test binary green (helpers, nf_table, whiten, phase_mask, phase_correct, nc_fastica, cebm, group, complex_io, pipeline).

- [ ] **Step 6: Commit**

```bash
git add rust/
git commit -m "feat(rust): end-to-end group complex ICA pipeline"
```

---

## Task 11: File-level driver

**Files:**
- Create: `rust/src/driver.rs`
- Modify: `rust/src/lib.rs` (add `pub mod driver;`)
- Create: `rust/tests/driver.rs`

**Interfaces:**
- Consumes: `complex_io` (Task 9), `pipeline::{run_complex_ica, Estimator, PipelineResult}` (Task 10).
- Produces:
  - `pub fn unmask(s: &DMatrix<Complex64>, mask: &[bool], dims: [usize; 3]) -> Vec<Vec<Complex64>>` — expands `(N, V_masked)` maps to `N` full-length volumes (zero outside the mask).
  - `pub fn run_from_files(pairs: &[(PathBuf, PathBuf)], n_components: usize, out_dir: &Path, kind: ComplexType, estimator: Estimator, nf_dir: &Path) -> Result<(PipelineResult, Vec<(PathBuf, PathBuf)>), String>` — loads each subject's two-file complex volume, flattens to `(T, V)`, runs the pipeline, un-masks the group maps to full volume shape, and writes one complex NIfTI pair per component. Returns the result and the written paths.

**Why this exists:** `run_complex_ica` is array-in/array-out and returns maps over masked voxels only. Without this layer a user has to re-implement the file → array → volume plumbing themselves. The Python port has the same driver (`run_from_files`), added for exactly this reason; keeping the two ports symmetric matters because they are meant to be interchangeable.

- [ ] **Step 1: Write the failing test**

`rust/tests/driver.rs`:

```rust
mod common;

use common::{fixtures_dir, TestRng};
use complex_gift::complex_io::{read_complex, write_complex, ComplexType};
use complex_gift::driver::{run_from_files, unmask};
use complex_gift::pipeline::Estimator;
use nalgebra::DMatrix;
use num_complex::Complex64;

#[test]
fn unmask_restores_full_volume_length_with_zeros_outside() {
    let dims = [4usize, 4, 2];
    let v = dims[0] * dims[1] * dims[2];
    let mask: Vec<bool> = (0..v).map(|i| i % 2 == 0).collect();
    let kept = mask.iter().filter(|&&m| m).count();

    let s = DMatrix::<Complex64>::from_fn(2, kept, |i, j| {
        Complex64::new((i * kept + j) as f64, 1.0)
    });
    let vols = unmask(&s, &mask, dims);

    assert_eq!(vols.len(), 2);
    assert_eq!(vols[0].len(), v);
    let mut c = 0usize;
    for j in 0..v {
        if mask[j] {
            assert!((vols[0][j] - s[(0, c)]).norm() < 1e-12);
            c += 1;
        } else {
            assert_eq!(vols[0][j], Complex64::new(0.0, 0.0)); // zero, not garbage
        }
    }
}

#[test]
fn run_from_files_writes_component_maps() {
    let mut rng = TestRng::new(51);
    let dims = [10usize, 10, 4]; // 400 voxels
    let v = dims[0] * dims[1] * dims[2];
    let (n, v_sig, t, n_sub) = (3usize, 300usize, 60usize, 3usize);

    let dir = std::env::temp_dir().join("cg_rust_driver");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let smaps = DMatrix::<f64>::from_fn(n, v_sig, |_, _| {
        rng.normal() * rng.normal().abs().powf(1.5)
    });
    let scale = smaps.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
    let smaps = smaps / scale;
    let phi: Vec<f64> = (0..v_sig)
        .map(|_| (rng.uniform() * 20.0 - 10.0) / 180.0 * std::f64::consts::PI)
        .collect();

    // write each subject as an R_/I_ pair holding a (dims, T) volume, flattened voxel-major
    let mut pairs = Vec::new();
    for s in 0..n_sub {
        let tc = DMatrix::<f64>::from_fn(t, n, |_, _| rng.normal());
        let modulation = &tc * &smaps;
        let mut flat = vec![Complex64::new(0.0, 0.0); v * t];
        for tt in 0..t {
            for vv in 0..v {
                let z = if vv < v_sig {
                    let mag = 100.0 + 10.0 * rng.uniform() + 5.0 * modulation[(tt, vv)];
                    Complex64::new(mag, 0.0) * Complex64::new(phi[vv].cos(), phi[vv].sin())
                } else {
                    let mag = 0.5 + rng.uniform();
                    let ang = std::f64::consts::TAU * rng.uniform();
                    Complex64::new(mag * ang.cos(), mag * ang.sin())
                };
                flat[tt * v + vv] = z; // volume-major per timepoint; the driver reshapes
            }
        }
        // write as T separate 3-D volumes is unnecessary here: write one 4-D pair by
        // delegating to write_complex per timepoint is awkward, so write the whole 4-D
        // stack through the same helper the driver reads back.
        let f1 = dir.join(format!("R_sub{s:02}.nii"));
        let f2 = dir.join(format!("I_sub{s:02}.nii"));
        write_4d(&flat, dims, t, &f1, &f2);
        pairs.push((f1, f2));
    }

    let out = dir.join("out");
    let (res, written) = run_from_files(
        &pairs,
        n,
        &out,
        ComplexType::RealImag,
        Estimator::Cebm { seed: 0 },
        &fixtures_dir(),
    )
    .expect("driver");

    let kept_sig = res.mask[..v_sig].iter().filter(|&&m| m).count() as f64 / v_sig as f64;
    assert!(kept_sig > 0.9, "mask dropped signal voxels: {kept_sig:.3}");

    assert_eq!(written.len(), n);
    for (f1, f2) in &written {
        assert!(f1.exists() && f2.exists());
        let (vals, d) = read_complex(f1, f2, ComplexType::RealImag).expect("read back");
        assert_eq!([d[0], d[1], d[2]], dims);
        assert_eq!(vals.len(), v);
        // voxels outside the mask are exactly zero in the written maps
        for j in 0..v {
            if !res.mask[j] {
                assert_eq!(vals[j], Complex64::new(0.0, 0.0));
            }
        }
        assert!(vals.iter().any(|z| z.norm() > 0.0), "empty component volume");
    }
}

/// Write a (V*T) voxel-major-per-timepoint buffer as a 4-D R_/I_ NIfTI pair.
fn write_4d(
    flat: &[Complex64],
    dims: [usize; 3],
    t: usize,
    first: &std::path::Path,
    second: &std::path::Path,
) {
    use nifti::writer::WriterOptions;
    let v = dims[0] * dims[1] * dims[2];
    let mut re = ndarray::Array4::<f64>::zeros((dims[0], dims[1], dims[2], t));
    let mut im = ndarray::Array4::<f64>::zeros((dims[0], dims[1], dims[2], t));
    for tt in 0..t {
        for vv in 0..v {
            // NIfTI is x-fastest; our flat voxel index runs the same way
            let x = vv % dims[0];
            let y = (vv / dims[0]) % dims[1];
            let z = vv / (dims[0] * dims[1]);
            re[(x, y, z, tt)] = flat[tt * v + vv].re;
            im[(x, y, z, tt)] = flat[tt * v + vv].im;
        }
    }
    WriterOptions::new(first).write_nifti(&re).unwrap();
    WriterOptions::new(second).write_nifti(&im).unwrap();
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd rust && cargo test --test driver 2>&1 | tail -10`
Expected: FAIL — unresolved import `complex_gift::driver`.

- [ ] **Step 3: Implement**

`rust/src/driver.rs`:

```rust
//! File-level driver: complex NIfTI in, complex component maps out.
//!
//! `pipeline::run_complex_ica` is array-in/array-out and returns maps over the masked
//! voxels only. This is the thin layer that makes the crate usable on real data: it loads
//! GIFT's two-file complex volumes, flattens them, runs the pipeline, expands the maps back
//! to full volume shape, and writes them out as complex NIfTI pairs.

use std::path::{Path, PathBuf};

use nalgebra::DMatrix;
use num_complex::Complex64;

use crate::complex_io::{read_complex, write_complex, ComplexType};
use crate::pipeline::{run_complex_ica, Estimator, PipelineResult};

/// Expand masked maps back to full volume length, zero outside the mask.
pub fn unmask(
    s: &DMatrix<Complex64>,
    mask: &[bool],
    dims: [usize; 3],
) -> Vec<Vec<Complex64>> {
    let v = dims[0] * dims[1] * dims[2];
    assert_eq!(mask.len(), v, "mask length does not match dims");
    let kept: Vec<usize> = (0..v).filter(|&j| mask[j]).collect();
    assert_eq!(kept.len(), s.ncols(), "maps do not match the mask");

    (0..s.nrows())
        .map(|i| {
            let mut vol = vec![Complex64::new(0.0, 0.0); v];
            for (c, &j) in kept.iter().enumerate() {
                vol[j] = s[(i, c)];
            }
            vol
        })
        .collect()
}

/// Run group complex ICA over subjects given as two-file complex NIfTI pairs, and write
/// the group component maps back out as complex NIfTI pairs.
pub fn run_from_files(
    pairs: &[(PathBuf, PathBuf)],
    n_components: usize,
    out_dir: &Path,
    kind: ComplexType,
    estimator: Estimator,
    nf_dir: &Path,
) -> Result<(PipelineResult, Vec<(PathBuf, PathBuf)>), String> {
    if pairs.is_empty() {
        return Err("run_from_files: no subjects".into());
    }

    let mut subjects = Vec::with_capacity(pairs.len());
    let mut dims3: Option<[usize; 3]> = None;

    for (first, second) in pairs {
        let (data, d) = read_complex(first, second, kind)?;
        let this = [d[0], d[1], d[2]];
        match dims3 {
            None => dims3 = Some(this),
            Some(prev) if prev != this => {
                return Err(format!("subjects disagree on volume shape: {prev:?} vs {this:?}"))
            }
            _ => {}
        }
        let v = this[0] * this[1] * this[2];
        let t = d[3];
        // read_complex returns the volume flattened x-fastest, with time slowest
        let mut m = DMatrix::<Complex64>::zeros(t, v);
        for tt in 0..t {
            for vv in 0..v {
                m[(tt, vv)] = data[tt * v + vv];
            }
        }
        subjects.push(m);
    }
    let dims = dims3.expect("at least one subject");

    let res = run_complex_ica(&subjects, n_components, None, estimator, nf_dir)?;

    let vols = unmask(&res.s_group, &res.mask, dims);
    std::fs::create_dir_all(out_dir)
        .map_err(|e| format!("cannot create {}: {e}", out_dir.display()))?;

    let mut written = Vec::with_capacity(vols.len());
    for (k, vol) in vols.iter().enumerate() {
        let base = format!("component_{:03}.nii", k + 1);
        let first = out_dir.join(format!("R_{base}"));
        let second = out_dir.join(format!("I_{base}"));
        write_complex(vol, dims, &first, &second, kind)?;
        written.push((first, second));
    }

    Ok((res, written))
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd rust && cargo test --test driver 2>&1 | tail -8`
Expected: `test result: ok. 2 passed`.

- [ ] **Step 5: Run the whole suite**

Run: `cd rust && cargo test 2>&1 | tail -20`
Expected: every test binary green.

- [ ] **Step 6: Commit**

```bash
git add rust/
git commit -m "feat(rust): file-level driver (complex NIfTI in -> component maps out)"
```

> **Phase 3 exit criterion (spec §9):** Rust matches Python (and, through it, MATLAB) on the
> shared fixtures — elementwise for the pure-arithmetic units, by ISI/correlation for the
> estimators; the phase-step property tests pass; the end-to-end group run recovers known
> sources; the file-level driver runs NIfTI in → component maps out. Deliverable: the
> `complex-gift` Rust crate.

---

## Notes for the implementer

- **Everything runs with plain `cargo`:** `cd rust && cargo test`. There is no LAPACK on this machine — `nalgebra` is pure Rust and needs no system libraries. Do NOT add `ndarray-linalg`; it cannot link here.
- **ICA cannot separate Gaussian sources.** Every source-recovery test must use non-Gaussian (super-Gaussian) sources. This is an identifiability limit, not a tuning problem — it has already cost a debugging cycle in an earlier phase.
- **Know what is comparable.** See the table in Global Constraints. Anything downstream of an eigendecomposition (whitening, both estimators, the group reduction) does NOT match Python elementwise — compare invariants or ISI. Anything that is pure arithmetic (ppval, Q, phase correction, back-reconstruction from fixed inputs) DOES — compare elementwise at 1e-10.
- **Memory order is a live trap.** `.npy` carries a `fortran_order` flag and npyz returns bytes as stored. Arrays that came through `scipy.io.loadmat` are **column-major** (cS/A/cX, the oracle W matrices, every nf*_coefs table); arrays built fresh in numpy are row-major. The loader dispatches on the flag. Guessing silently transposes the fixture - no error, just wrong numbers.
- **Tasks 6 and 7 are translations** of already-validated Python, not fresh algorithm design. Read the Python (and the MATLAB it came from) completely before writing Rust; the tests are the executable specification.
- **The fixtures are read-only.** If a test seems to want different fixture data, the test is wrong.
