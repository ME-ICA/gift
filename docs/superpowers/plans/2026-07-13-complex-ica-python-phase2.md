# Complex ICA — Python Port (`complex-gift`, Phase 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Python package `complex-gift` that runs the complete complex-valued fMRI ICA pipeline (complex I/O → phase-quality mask → complex whitening → complex ICA → phase-ambiguity correction → group back-reconstruction), validated against the MATLAB oracle produced in Phase 0+1.

**Architecture:** One module per pipeline unit from spec §4, mirroring the MATLAB implementation so tests transfer. Two estimators behind a small interface: nc-FastICA (deterministic) and Complex ICA-EBM (stochastic). The MATLAB artifacts in `complex_ica_fixtures/` are the read-only correctness oracle.

**Tech Stack:** Python 3.12, NumPy 2.5, SciPy 1.18, nibabel 5.4, scikit-image 0.26, pytest 9.1 — all in the micromamba env **`giftenv`**.

## Global Constraints

- **Environment:** every command runs in `giftenv`. Prefix with `micromamba run -n giftenv …`. Never install into `base`; never create new envs.
- **License:** GPL v3. `estimators/cebm.py` and `estimators/nc_fastica.py` are translations of GPL-v3 Adalí-lab MATLAB — each carries a GPL v3 header naming the source file and paper.
- **Python project root is `python/`.** Package `complex_gift/`, tests `python/tests/`. Do not put a `pyproject.toml` at the repo root (this is primarily a MATLAB repo).
- **dtype:** `np.complex128` throughout. Never silently downcast to real.
- **Orientation:** `X` is `(N, T)` = components/channels × samples, matching the MATLAB convention. For spatial fMRI ICA, `T` is voxels.
- **Fixtures are READ-ONLY.** `complex_ica_fixtures/` (repo root) is the committed MATLAB oracle. Never regenerate, overwrite, or hand-edit it.
- **MATLAB→NumPy translation hazards** (apply to every ported file):
  - MATLAB `'` is the **conjugate** transpose (`.conj().T`); `.'` is the plain transpose (`.T`). Getting this wrong is the single most likely silent bug.
  - MATLAB is 1-indexed; Python is 0-indexed.
  - MATLAB `round()` rounds half **away from zero**; Python's `round()` is banker's rounding. Use `int(math.floor(x + 0.5))` where a MATLAB `round` on a positive value is being ported.
  - MATLAB `eig` on a Hermitian matrix ↔ `numpy.linalg.eigh` (eigenvalue **order and eigenvector sign/phase differ** from MATLAB — never depend on them).

---

## Task 1: Package scaffolding + oracle test helpers

**Files:**
- Create: `python/pyproject.toml`
- Create: `python/complex_gift/__init__.py`
- Create: `python/tests/conftest.py`
- Create: `python/tests/oracle.py`
- Create: `python/tests/test_oracle_helpers.py`
- Create: `CLAUDE.md` (repo root — records the env for future sessions)

**Interfaces:**
- Produces: `tests/oracle.py::isi(G) -> float` (Amari ISI of a global matrix; 0 = perfect separation) and `tests/oracle.py::match_sources(S_est, S_true) -> (perm, corr)` (match each estimated source to a true source by |complex correlation|). Both are used by every estimator test.
- Produces: pytest fixture `fixtures_dir` (pathlib.Path to repo-root `complex_ica_fixtures/`) and `sources` (dict from `complex_sources.mat` with keys `cS`, `A`, `cX`).

- [ ] **Step 1: Create the package skeleton**

`python/pyproject.toml`:

```toml
[project]
name = "complex-gift"
version = "0.1.0"
description = "Complex-valued fMRI ICA - Python port of GIFT's complex ICA pipeline"
requires-python = ">=3.11"
license = { text = "GPL-3.0-or-later" }
dependencies = [
    "numpy>=2.0",
    "scipy>=1.13",
    "nibabel>=5.2",
    "scikit-image>=0.22",
]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
include = ["complex_gift*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`python/complex_gift/__init__.py`:

```python
"""complex-gift: complex-valued fMRI ICA (Python port of GIFT's complex ICA pipeline).

GPL v3 - derives from the Adali-lab (MLSP/UMBC) complex ICA algorithms.
"""

__version__ = "0.1.0"
```

`CLAUDE.md` (repo root):

```markdown
# GIFT

MATLAB toolbox (GroupICAT/) plus a Python port under `python/` (`complex-gift`).

## Environment

The Python port uses the micromamba environment **`giftenv`**
(python 3.12, numpy, scipy, nibabel, scikit-image, pytest).

Run everything through it:

    micromamba run -n giftenv pytest python/tests -q

## Complex ICA

- Design spec: `docs/superpowers/specs/2026-07-13-complex-ica-port-design.md`
- Developer reference: `doc/complex_ica_porting_reference.md`
- MATLAB oracle artifacts: `complex_ica_fixtures/` (READ-ONLY - generated once from MATLAB)
```

- [ ] **Step 2: Write the failing test for the oracle helpers**

`python/tests/oracle.py` will hold the helpers; `python/tests/test_oracle_helpers.py`:

```python
import numpy as np
from oracle import isi, match_sources


def test_isi_zero_for_identity_and_scaled_permutation():
    assert isi(np.eye(4)) == 0.0
    # a scaled permutation is perfect separation too
    G = np.zeros((4, 4), dtype=complex)
    perm = [2, 0, 3, 1]
    for i, j in enumerate(perm):
        G[i, j] = (i + 1) * np.exp(1j * 0.3 * i)
    assert isi(G) < 1e-12


def test_isi_positive_for_mixed_matrix():
    G = np.ones((4, 4), dtype=complex)   # maximally mixed
    assert isi(G) > 0.9


def test_match_sources_recovers_permutation_and_phase():
    rng = np.random.default_rng(0)
    S = rng.standard_normal((3, 500)) + 1j * rng.standard_normal((3, 500))
    perm_true = [1, 2, 0]
    # permuted + arbitrarily phase-scaled copies
    S_est = np.array([S[j] * (2.0 * np.exp(1j * 0.7 * k)) for k, j in enumerate(perm_true)])
    perm, corr = match_sources(S_est, S)
    assert list(perm) == perm_true
    assert np.all(corr > 0.99)
```

- [ ] **Step 3: Run to verify it fails**

Run: `micromamba run -n giftenv pytest python/tests/test_oracle_helpers.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'oracle'`.

- [ ] **Step 4: Implement the helpers and fixtures**

`python/tests/oracle.py`:

```python
"""Correctness-oracle helpers shared by the estimator tests."""

import numpy as np


def isi(G):
    """Amari inter-symbol interference of a global matrix G = W @ A.

    0.0 == perfect separation (G is a scaled permutation). Scale- and
    permutation-invariant, which is exactly the indeterminacy of ICA.
    """
    G = np.abs(np.asarray(G))
    n = G.shape[0]
    row = (G / G.max(axis=1, keepdims=True)).sum(axis=1) - 1.0
    col = (G / G.max(axis=0, keepdims=True)).sum(axis=0) - 1.0
    return float((row.sum() + col.sum()) / (2 * n * (n - 1)))


def match_sources(S_est, S_true):
    """Match each estimated source to a true source by |complex correlation|.

    Returns (perm, corr): perm[k] is the index of the true source best matching
    estimated source k; corr[k] is that |correlation| in [0, 1]. Invariant to the
    per-source complex scale/phase ambiguity of complex ICA.
    """
    S_est = np.asarray(S_est)
    S_true = np.asarray(S_true)
    Se = S_est - S_est.mean(axis=1, keepdims=True)
    St = S_true - S_true.mean(axis=1, keepdims=True)
    Se = Se / np.linalg.norm(Se, axis=1, keepdims=True)
    St = St / np.linalg.norm(St, axis=1, keepdims=True)
    C = np.abs(Se @ St.conj().T)          # (n_est, n_true)
    perm = C.argmax(axis=1)
    return perm, C.max(axis=1)
```

`python/tests/conftest.py`:

```python
import sys
from pathlib import Path

import pytest
from scipy.io import loadmat

# make tests/oracle.py importable as `oracle`
sys.path.insert(0, str(Path(__file__).parent))

FIXTURES = Path(__file__).resolve().parents[2] / "complex_ica_fixtures"


@pytest.fixture(scope="session")
def fixtures_dir():
    assert FIXTURES.is_dir(), f"MATLAB oracle fixtures not found at {FIXTURES}"
    return FIXTURES


@pytest.fixture(scope="session")
def sources(fixtures_dir):
    """Ground truth from MATLAB: cX = A @ cS."""
    d = loadmat(fixtures_dir / "complex_sources.mat")
    return {"cS": d["cS"], "A": d["A"], "cX": d["cX"]}
```

- [ ] **Step 5: Run to verify it passes**

Run: `micromamba run -n giftenv pytest python/tests/test_oracle_helpers.py -q`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add python/pyproject.toml python/complex_gift/__init__.py python/tests/ CLAUDE.md
git commit -m "feat(py): complex-gift scaffolding + ISI/source-matching oracle helpers"
```

---

## Task 2: `nf_table` loader, `simplified_ppval`, and the canonical portable export

**Files:**
- Create: `python/complex_gift/nf_table.py`
- Create: `python/tests/test_nf_table.py`
- Output (git-tracked): `complex_ica_fixtures/nf_table.npz` (canonical export consumed by the Rust track)

**Interfaces:**
- Consumes: `complex_ica_fixtures/nf_table.mat` (MATLAB structs; verified readable via `scipy.io.loadmat`).
- Produces:
  - `PP` — dataclass with `breaks: np.ndarray (41,)`, `coefs: np.ndarray (40, 4)`, `pieces: int`, `order: int`.
  - `NF` — dataclass with `min_EGx: float`, `max_EGx: float`, `critical_point: float`, `critical_point2: float`, `pp: PP`, `pp_slope: PP`.
  - `load_nf_table(path) -> dict[str, NF]` keyed `"nf1".."nf8"`.
  - `simplified_ppval(pp: PP, xs: float) -> float` — an exact port of `simplified_ppval` in `complex_ICA_EBM.m:755-802`.
  - `export_nf_table(mat_path, npz_path) -> None` — writes the language-neutral `.npz` for Rust.

**Background (already verified):** each `nfK` is a MATLAB struct with the four scalars plus `pp`/`pp_slope`, which are standard MATLAB piecewise-polynomial forms (`breaks` 41 knots, `coefs` 40×4, cubic, `pieces=40`, `order=4`).

- [ ] **Step 1: Write the failing test**

`python/tests/test_nf_table.py`:

```python
import numpy as np
import pytest
from scipy.interpolate import PPoly

from complex_gift.nf_table import export_nf_table, load_nf_table, simplified_ppval


@pytest.fixture(scope="module")
def nf(fixtures_dir):
    return load_nf_table(fixtures_dir / "nf_table.mat")


def test_structure(nf):
    assert sorted(nf) == [f"nf{i}" for i in range(1, 9)]
    for k, v in nf.items():
        assert v.pp.coefs.shape == (v.pp.pieces, 4)
        assert v.pp.breaks.shape == (v.pp.pieces + 1,)
        assert v.pp.order == 4
        assert v.max_EGx > v.min_EGx


def test_ppval_matches_scipy_ppoly(nf):
    """Cross-check the hand-ported evaluator against an independent implementation.

    MATLAB pp: on [breaks[i], breaks[i+1]) the value is
        sum_j coefs[i, j] * (x - breaks[i])**(order-1-j)
    which is exactly scipy PPoly with c = coefs.T. Two independent evaluators
    agreeing is strong evidence the port of simplified_ppval is faithful.
    """
    for v in nf.values():
        for pp in (v.pp, v.pp_slope):
            ref = PPoly(c=pp.coefs.T.copy(), x=pp.breaks.copy())
            xs = np.linspace(pp.breaks[0], pp.breaks[-1], 97)[1:-1]
            got = np.array([simplified_ppval(pp, float(x)) for x in xs])
            assert np.allclose(got, ref(xs), rtol=1e-10, atol=1e-12)


def test_ppval_clamps_outside_breaks(nf):
    """Outside the knot range MATLAB's evaluator extrapolates from the end piece
    (index clamped to first/last), rather than raising."""
    pp = nf["nf1"].pp
    assert np.isfinite(simplified_ppval(pp, float(pp.breaks[0] - 5.0)))
    assert np.isfinite(simplified_ppval(pp, float(pp.breaks[-1] + 5.0)))


def test_export_npz_roundtrip(nf, fixtures_dir, tmp_path):
    out = tmp_path / "nf_table.npz"
    export_nf_table(fixtures_dir / "nf_table.mat", out)
    d = np.load(out)
    for name, v in nf.items():
        assert np.array_equal(d[f"{name}/pp/breaks"], v.pp.breaks)
        assert np.array_equal(d[f"{name}/pp/coefs"], v.pp.coefs)
        assert np.array_equal(d[f"{name}/pp_slope/coefs"], v.pp_slope.coefs)
        assert float(d[f"{name}/min_EGx"]) == v.min_EGx
        assert float(d[f"{name}/critical_point2"]) == v.critical_point2
```

- [ ] **Step 2: Run to verify it fails**

Run: `micromamba run -n giftenv pytest python/tests/test_nf_table.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'complex_gift.nf_table'`.

- [ ] **Step 3: Implement the loader + evaluator + export**

`python/complex_gift/nf_table.py`:

```python
"""Nonlinearity lookup tables for Complex ICA-EBM.

The tables ship as MATLAB structs holding piecewise-polynomial (spline) forms.
`simplified_ppval` is an exact port of the evaluator bundled with the reference
implementation (complex_ICA_EBM.m:755-802) - ported rather than replaced with a
library spline so the numerics match the reference exactly.

GPL v3 - derives from the Adali-lab (MLSP/UMBC) complex ICA-EBM implementation.
Reference: Li & Adali (2010), IEEE Trans. Circuits Syst. I, 57(7):1417-1430.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.io import loadmat

_NF_NAMES = [f"nf{i}" for i in range(1, 9)]


@dataclass(frozen=True)
class PP:
    """MATLAB piecewise-polynomial form."""

    breaks: np.ndarray   # (pieces + 1,)
    coefs: np.ndarray    # (pieces, order), descending powers of (x - breaks[i])
    pieces: int
    order: int


@dataclass(frozen=True)
class NF:
    """One entropy-bound nonlinearity: scalars + its spline and the spline's slope.

    NOTE: `critical_point2` is present only on nf1 in the source table; it is NaN for
    nf2-nf8 by design. It is inert legacy metadata - the reference complex_ICA_EBM.m
    never reads it. Do not treat the NaN as a loader bug.
    """

    min_EGx: float
    max_EGx: float
    critical_point: float
    critical_point2: float
    pp: PP
    pp_slope: PP


def _to_pp(m) -> PP:
    return PP(
        breaks=np.asarray(m.breaks, dtype=np.float64).ravel(),
        coefs=np.asarray(m.coefs, dtype=np.float64),
        pieces=int(m.pieces),
        order=int(m.order),
    )


def load_nf_table(path) -> dict[str, NF]:
    """Load nf_table.mat / complex_nf_table.mat into plain Python objects."""
    d = loadmat(path, squeeze_me=True, struct_as_record=False)
    out = {}
    for name in _NF_NAMES:
        m = d[name]
        out[name] = NF(
            min_EGx=float(m.min_EGx),
            max_EGx=float(m.max_EGx),
            critical_point=float(m.critical_point),
            # only nf1 carries critical_point2; nf2-nf8 genuinely lack the field
            critical_point2=float(getattr(m, "critical_point2", np.nan)),
            pp=_to_pp(m.pp),
            pp_slope=_to_pp(m.pp_slope),
        )
    return out


def simplified_ppval(pp: PP, xs: float) -> float:
    """Exact port of the reference's simplified_ppval (complex_ICA_EBM.m:755-802).

    Binary-searches for the piece, shifts to local coordinates, then evaluates by
    nested (Horner) multiplication. NOTE: the reference's `round((low+high)/2)` uses
    MATLAB rounding (half away from zero); Python's round() is banker's rounding, so
    floor(x + 0.5) is used here to reproduce it exactly.
    """
    b = pp.breaks
    c = pp.coefs
    ell = pp.pieces          # MATLAB `l`
    k = 4                    # the reference hardcodes order 4

    # --- find the piece index (MATLAB 1-based; converted to 0-based at the end) ---
    if xs > b[ell - 1]:            # MATLAB: xs > b(l)
        index = ell               # MATLAB: index = l
    elif xs < b[1]:               # MATLAB: xs < b(2)
        index = 1                 # MATLAB: index = 1
    else:
        low_index = 1
        high_index = ell
        while True:
            middle_index = int(math.floor((low_index + high_index) / 2 + 0.5))
            if b[middle_index - 1] > xs:
                high_index = middle_index
            else:
                low_index = middle_index
            if low_index == high_index - 1:
                index = low_index
                break

    i = index - 1                 # to 0-based row of coefs / breaks

    # --- local coordinates, then nested multiplication ---
    xs = xs - b[i]
    v = c[i, 0]
    for j in range(1, k):
        v = xs * v + c[i, j]
    return float(v)


def export_nf_table(mat_path, npz_path) -> None:
    """Write the language-neutral canonical export consumed by the Rust port."""
    nf = load_nf_table(mat_path)
    flat = {}
    for name, v in nf.items():
        flat[f"{name}/min_EGx"] = np.float64(v.min_EGx)
        flat[f"{name}/max_EGx"] = np.float64(v.max_EGx)
        flat[f"{name}/critical_point"] = np.float64(v.critical_point)
        flat[f"{name}/critical_point2"] = np.float64(v.critical_point2)
        for attr in ("pp", "pp_slope"):
            pp = getattr(v, attr)
            flat[f"{name}/{attr}/breaks"] = pp.breaks
            flat[f"{name}/{attr}/coefs"] = pp.coefs
    np.savez(npz_path, **flat)
```

- [ ] **Step 4: Run to verify it passes**

Run: `micromamba run -n giftenv pytest python/tests/test_nf_table.py -q`
Expected: 4 passed.

- [ ] **Step 5: Generate the committed canonical export**

Run:
```bash
micromamba run -n giftenv python -c "
from complex_gift.nf_table import export_nf_table
export_nf_table('../complex_ica_fixtures/nf_table.mat', '../complex_ica_fixtures/nf_table.npz')
print('wrote nf_table.npz')
" 
```
(run from `python/`). Expected: `wrote nf_table.npz`.

- [ ] **Step 6: Commit**

```bash
git add python/complex_gift/nf_table.py python/tests/test_nf_table.py complex_ica_fixtures/nf_table.npz
git commit -m "feat(py): nf_table loader, exact simplified_ppval port, canonical npz export"
```

---

## Task 3: Complex whitening (Hermitian + strong uncorrelating transform)

**Files:**
- Create: `python/complex_gift/whiten.py`
- Create: `python/tests/test_whiten.py`

**Interfaces:**
- Produces:
  - `whiten_hermitian(X, n_components=None) -> (Xw, W_whiten, W_dewhiten)` — mirrors `icatb_pca_whitening.m`. `X` is `(P, T)` complex; `Xw` is `(N, T)` with `Xw @ Xw.conj().T / T ≈ I`; `W_whiten` is `(N, P)`; `W_dewhiten` is `(P, N)`.
  - `strong_uncorrelating_transform(X) -> (Xs, W_sut)` — simultaneously whitens the covariance `E[xx^H]` and diagonalizes the pseudo-covariance `E[xx^T]` (needed by noncircular estimators).

- [ ] **Step 1: Write the failing test**

`python/tests/test_whiten.py`:

```python
import numpy as np

from complex_gift.whiten import strong_uncorrelating_transform, whiten_hermitian


def test_hermitian_whitening_decorrelates():
    rng = np.random.default_rng(1)
    P, T, N = 8, 4000, 4
    S = rng.standard_normal((N, T)) + 1j * rng.standard_normal((N, T))
    Amix = rng.standard_normal((P, N)) + 1j * rng.standard_normal((P, N))
    X = Amix @ S
    Xw, W_wh, W_dw = whiten_hermitian(X, n_components=N)
    assert Xw.shape == (N, T)
    assert Xw.dtype == np.complex128
    C = Xw @ Xw.conj().T / T
    assert np.allclose(C, np.eye(N), atol=1e-6)


def test_dewhitening_reconstructs_signal_subspace():
    rng = np.random.default_rng(2)
    P, T, N = 6, 3000, 3
    S = rng.standard_normal((N, T)) + 1j * rng.standard_normal((N, T))
    Amix = rng.standard_normal((P, N)) + 1j * rng.standard_normal((P, N))
    X = Amix @ S
    Xc = X - X.mean(axis=1, keepdims=True)
    Xw, W_wh, W_dw = whiten_hermitian(X, n_components=N)
    # data is exactly rank N, so dewhitening must reconstruct it
    assert np.allclose(W_dw @ Xw, Xc, atol=1e-6)


def test_sut_diagonalizes_covariance_and_pseudo_covariance():
    rng = np.random.default_rng(3)
    N, T = 4, 20000
    # noncircular sources: unequal real/imag variance => nonzero pseudo-covariance
    S = rng.standard_normal((N, T)) + 1j * 0.3 * rng.standard_normal((N, T))
    Amix = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    X = Amix @ S
    Xs, W_sut = strong_uncorrelating_transform(X)
    C = Xs @ Xs.conj().T / T          # covariance -> identity
    Pc = Xs @ Xs.T / T                # pseudo-covariance -> diagonal (real, nonneg)
    assert np.allclose(C, np.eye(N), atol=1e-2)
    off = Pc - np.diag(np.diag(Pc))
    assert np.abs(off).max() < 1e-2 * max(1.0, np.abs(np.diag(Pc)).max())
```

- [ ] **Step 2: Run to verify it fails**

Run: `micromamba run -n giftenv pytest python/tests/test_whiten.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'complex_gift.whiten'`.

- [ ] **Step 3: Implement**

`python/complex_gift/whiten.py`:

```python
"""Complex whitening.

`whiten_hermitian` mirrors GIFT's icatb_pca_whitening.m: it uses only the
covariance E[xx^H], which is the right thing for circular (proper) sources.

`strong_uncorrelating_transform` additionally diagonalizes the pseudo-covariance
E[xx^T], which carries signal when sources are noncircular - required by
noncircular estimators such as nc-FastICA.
"""

import numpy as np
import scipy.linalg as sla


def whiten_hermitian(X, n_components=None):
    """PCA-whiten complex data using the Hermitian covariance.

    X: (P, T) complex. Returns (Xw (N, T), W_whiten (N, P), W_dewhiten (P, N)).
    """
    X = np.asarray(X, dtype=np.complex128)
    P, T = X.shape
    N = P if n_components is None else int(n_components)

    Xc = X - X.mean(axis=1, keepdims=True)
    R = Xc @ Xc.conj().T / (T - 1)          # (P, P) Hermitian; note conjugate transpose
    d, U = np.linalg.eigh(R)                # ascending, real eigenvalues
    order = np.argsort(d)[::-1][:N]         # descending, keep top N
    d = d[order].real
    U = U[:, order]

    s = np.sqrt(d)
    W_whiten = (U.conj().T) / s[:, None]    # (N, P)
    W_dewhiten = U * s[None, :]             # (P, N)
    Xw = W_whiten @ Xc
    return Xw, W_whiten, W_dewhiten


def strong_uncorrelating_transform(X):
    """Simultaneously whiten E[xx^H] and diagonalize the pseudo-covariance E[xx^T].

    X: (N, T) complex. Returns (Xs (N, T), W_sut (N, N)).
    """
    X = np.asarray(X, dtype=np.complex128)
    N, T = X.shape
    Xc = X - X.mean(axis=1, keepdims=True)

    # 1. standard Hermitian whitening
    Xw, W_wh, _ = whiten_hermitian(Xc, n_components=N)

    # 2. the whitened pseudo-covariance is complex symmetric; its Takagi
    #    factorization P = U diag(k) U^T gives the rotation that diagonalizes it
    #    while preserving E[xx^H] = I (U is unitary).
    Pc = Xw @ Xw.T / T
    # Takagi via SVD of a complex symmetric matrix: P = V S V^T
    U_, S_, Vh_ = np.linalg.svd(Pc)
    # for complex symmetric P, the Takagi vectors follow from a phase correction
    Z = U_.conj().T @ np.conj(Vh_.conj().T)
    w, Q = sla.schur(Z, output="complex")
    phase = np.sqrt(np.diag(w))
    V = U_ @ Q @ np.diag(phase)
    W_sut = V.conj().T @ W_wh
    Xs = W_sut @ Xc
    return Xs, W_sut
```

- [ ] **Step 4: Run to verify it passes**

Run: `micromamba run -n giftenv pytest python/tests/test_whiten.py -q`
Expected: 3 passed.

> If `test_sut_diagonalizes_covariance_and_pseudo_covariance` fails, the Takagi
> factorization is the suspect (it is the only subtle step). Debug it directly:
> the requirement is `V` unitary with `V.conj().T @ Pc @ V.conj()` real, nonnegative,
> diagonal. Do not weaken the test to make it pass.

- [ ] **Step 5: Commit**

```bash
git add python/complex_gift/whiten.py python/tests/test_whiten.py
git commit -m "feat(py): complex Hermitian whitening + strong uncorrelating transform"
```

---

## Task 4: Phase-quality mask

**Files:**
- Create: `python/complex_gift/phase_mask.py`
- Create: `python/tests/test_phase_mask.py`

**Interfaces:**
- Produces:
  - `quality_map(Z) -> Q` — `Q[v] = |sum_t Z[v, t]| / sum_t |Z[v, t]|`, in `[0, 1]`. `Z` is `(V, T)` complex.
  - `otsu_threshold(x) -> float`
  - `phase_quality_mask(Z, mag_mask=None) -> (mask, Q, tau)` — `mask = mag_mask & (Q > tau)`, `tau` from Otsu over the in-mask voxels.
- Mirrors the MATLAB `icatb_complex_phase_mask` (same formula, same properties).

**Key property (this is the point of the unit):** `Q` is invariant to a constant per-voxel phase offset (multiplying a voxel's whole time series by a fixed `e^{i phi}`), because both `|sum_t z|` and `sum_t |z|` are unchanged. That is what lets it run on raw complex data before any background-phase removal.

- [ ] **Step 1: Write the failing property test**

`python/tests/test_phase_mask.py`:

```python
import numpy as np

from complex_gift.phase_mask import otsu_threshold, phase_quality_mask, quality_map


def _synthetic(rng, n_stable=250, n_random=250, T=80):
    """Stable-phase voxels (signal) and random-phase voxels (noise)."""
    V = n_stable + n_random
    mag = 1.0 + 0.1 * rng.standard_normal((V, T))
    phi = np.empty((V, T))
    phi[:n_stable] = 0.1 * rng.standard_normal((n_stable, T))     # stable
    phi[n_stable:] = 2 * np.pi * rng.random((n_random, T))        # random
    return mag * np.exp(1j * phi), n_stable


def test_quality_map_separates_stable_from_random_phase():
    rng = np.random.default_rng(3)
    Z, n = _synthetic(rng)
    Q = quality_map(Z)
    assert np.all((Q >= 0) & (Q <= 1))
    assert Q[:n].mean() > Q[n:].mean() + 0.3


def test_mask_selects_stable_voxels():
    rng = np.random.default_rng(3)
    Z, n = _synthetic(rng)
    mask, Q, tau = phase_quality_mask(Z)
    assert mask[:n].mean() > 0.8
    assert mask[n:].mean() < 0.2


def test_invariance_to_constant_per_voxel_phase():
    """THE defining property: a static per-voxel phase offset (B0/receiver phase)
    must not change Q or the mask at all."""
    rng = np.random.default_rng(3)
    Z, _ = _synthetic(rng)
    offset = np.exp(1j * 2 * np.pi * rng.random((Z.shape[0], 1)))  # constant over time
    mask1, Q1, _ = phase_quality_mask(Z)
    mask2, Q2, _ = phase_quality_mask(Z * offset)
    assert np.abs(Q1 - Q2).max() < 1e-10
    assert np.array_equal(mask1, mask2)


def test_otsu_splits_a_bimodal_distribution():
    rng = np.random.default_rng(4)
    x = np.concatenate([rng.normal(0.1, 0.02, 500), rng.normal(0.9, 0.02, 500)])
    tau = otsu_threshold(x)
    assert 0.2 < tau < 0.8
```

- [ ] **Step 2: Run to verify it fails**

Run: `micromamba run -n giftenv pytest python/tests/test_phase_mask.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'complex_gift.phase_mask'`.

- [ ] **Step 3: Implement**

`python/complex_gift/phase_mask.py`:

```python
"""Phase-quality mask (quality-map thresholding).

Phase is only trustworthy where SNR is high: in brain voxels the complex time
series points in a nearly consistent direction over time, while in noise voxels the
phase wanders. Temporal phase stability is therefore a proxy for SNR.

Reference: Rodriguez, Correa, Eichele, Calhoun & Adali (2011), J. Signal Process.
Syst. 65:497-508.
"""

import numpy as np
from skimage.filters import threshold_otsu


def quality_map(Z):
    """Q[v] = |sum_t Z[v, t]| / sum_t |Z[v, t]|, in [0, 1]. Z is (V, T) complex.

    Invariant to a constant per-voxel phase offset (it measures variation over time).
    """
    Z = np.asarray(Z, dtype=np.complex128)
    num = np.abs(Z.sum(axis=1))
    den = np.abs(Z).sum(axis=1) + np.finfo(np.float64).eps
    return num / den


def otsu_threshold(x):
    """Otsu threshold of a 1-D array."""
    return float(threshold_otsu(np.asarray(x, dtype=np.float64)))


def phase_quality_mask(Z, mag_mask=None):
    """(mask, Q, tau). Intersects a magnitude/brain mask with Q > Otsu(Q)."""
    Z = np.asarray(Z, dtype=np.complex128)
    V = Z.shape[0]
    if mag_mask is None:
        mag_mask = np.ones(V, dtype=bool)
    mag_mask = np.asarray(mag_mask, dtype=bool).ravel()

    Q = quality_map(Z)
    tau = otsu_threshold(Q[mag_mask])
    mask = mag_mask & (Q > tau)
    return mask, Q, tau
```

- [ ] **Step 4: Run to verify it passes**

Run: `micromamba run -n giftenv pytest python/tests/test_phase_mask.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add python/complex_gift/phase_mask.py python/tests/test_phase_mask.py
git commit -m "feat(py): phase-quality mask (quality map + Otsu)"
```

---

## Task 5: Phase-ambiguity correction and group phase alignment

**Files:**
- Create: `python/complex_gift/phase_correct.py`
- Create: `python/tests/test_phase_correct.py`

**Interfaces:**
- Produces:
  - `correct_phase(S, A, mask=None) -> (S, A, theta)` — `S` is `(N, V)` complex sources, `A` is `(M, N)` complex mixing. Per component `k`: `theta[k] = -0.5 * angle(sum(S[k, mask]**2))`, then `S[k] *= exp(1j*theta[k])` and `A[:, k] *= exp(-1j*theta[k])` (so `A @ S` is preserved), then a residual pi sign fix making the real-part skewness positive.
  - `align_to_reference(S, S_ref) -> (S, theta)` — group alignment (spec §4 `group` unit): `theta[k] = -angle(sum(conj(S_ref[k]) * S[k]))`, `S[k] *= exp(1j*theta[k])`.

**Why:** complex ICA recovers each source only up to a complex scalar `c_k e^{i theta_k}`, so the split of a component's energy between real and imaginary parts is arbitrary and components aren't comparable across subjects until `theta_k` is fixed by convention. The convention: rotate so energy is maximally concentrated in the real part. Squaring maps the +/-alpha line ambiguity to a single angle, hence the closed form. Reference: Rodriguez, Calhoun & Adali (2012), Pattern Recognition 45:2050-2063.

- [ ] **Step 1: Write the failing property test**

`python/tests/test_phase_correct.py`:

```python
import numpy as np

from complex_gift.phase_correct import align_to_reference, correct_phase


def _aligned_sources(rng, N=4, V=3000):
    """Sources whose energy already lies along the real axis."""
    return rng.standard_normal((N, V)) + 1j * 0.05 * rng.standard_normal((N, V))


def test_reconstruction_is_preserved():
    """The whole point: rotating S by e^{i0} and A by e^{-i0} must leave A @ S alone."""
    rng = np.random.default_rng(5)
    N, V, M = 4, 3000, 20
    S0 = _aligned_sources(rng, N, V)
    A0 = rng.standard_normal((M, N)) + 1j * rng.standard_normal((M, N))
    theta_true = np.array([0.7, -1.1, 0.3, 2.0])
    S = S0 * np.exp(1j * theta_true)[:, None]
    A = A0 * np.exp(-1j * theta_true)[None, :]

    X_before = A @ S
    Sc, Ac, theta = correct_phase(S, A)
    rel = np.linalg.norm(X_before - Ac @ Sc) / np.linalg.norm(X_before)
    assert rel < 1e-10


def test_injected_rotation_is_removed():
    """A known rotation is undone: corrected sources concentrate on the real axis."""
    rng = np.random.default_rng(5)
    N, V, M = 4, 3000, 20
    S0 = _aligned_sources(rng, N, V)
    A0 = rng.standard_normal((M, N)) + 1j * rng.standard_normal((M, N))
    theta_true = np.array([0.7, -1.1, 0.3, 2.0])
    S = S0 * np.exp(1j * theta_true)[:, None]
    A = A0 * np.exp(-1j * theta_true)[None, :]

    Sc, Ac, theta = correct_phase(S, A)
    imag_frac = (Sc.imag**2).sum(axis=1) / (np.abs(Sc) ** 2).sum(axis=1)
    assert np.all(imag_frac < 0.05)
    # residual major-axis angle is ~0 mod pi
    resid = np.mod(np.angle((Sc**2).sum(axis=1)), np.pi)
    assert np.all(np.minimum(resid, np.pi - resid) < 1e-3)


def test_mask_restricts_the_estimate():
    """theta is estimated only from masked (high-quality) voxels: garbage outside the
    mask must not move the answer."""
    rng = np.random.default_rng(6)
    N, V, M = 3, 2000, 10
    S = _aligned_sources(rng, N, V)
    A = rng.standard_normal((M, N)) + 1j * rng.standard_normal((M, N))
    mask = np.zeros(V, dtype=bool)
    mask[: V // 2] = True
    S_dirty = S.copy()
    S_dirty[:, ~mask] *= 50.0 * np.exp(1j * 1.3)      # wild values outside the mask

    _, _, theta_clean = correct_phase(S.copy(), A.copy(), mask=mask)
    _, _, theta_dirty = correct_phase(S_dirty, A.copy(), mask=mask)
    assert np.allclose(theta_clean, theta_dirty, atol=1e-9)


def test_group_alignment_collapses_known_subject_rotations():
    rng = np.random.default_rng(7)
    N, V = 3, 1500
    S_ref = _aligned_sources(rng, N, V)
    rot = np.array([0.9, -0.4, 2.2])
    S_subj = S_ref * np.exp(1j * rot)[:, None]
    S_al, theta = align_to_reference(S_subj, S_ref)
    assert np.allclose(np.abs(theta), np.abs(rot), atol=1e-6)
    # aligned sources now agree with the reference up to a positive real scale
    for k in range(N):
        c = np.vdot(S_ref[k], S_al[k]) / np.vdot(S_ref[k], S_ref[k])
        assert abs(c.imag) < 1e-6
```

- [ ] **Step 2: Run to verify it fails**

Run: `micromamba run -n giftenv pytest python/tests/test_phase_correct.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'complex_gift.phase_correct'`.

- [ ] **Step 3: Implement**

`python/complex_gift/phase_correct.py`:

```python
"""Phase-ambiguity correction (the complex-domain analogue of real ICA's sign fix).

Complex ICA recovers each source only up to a complex scalar c_k * exp(i*theta_k):

    X = A @ S  <=>  X = (A @ inv(D)) @ (D @ S),   D = diag(c_k * exp(i*theta_k))

so the split of a component's energy between real and imaginary parts is arbitrary.
Convention: rotate each component so its energy is maximally concentrated in the real
part. The voxel values form an elongated cloud ~ r(v) * exp(i*alpha); squaring maps the
+/-alpha line ambiguity to the single angle 2*alpha, giving the closed form
theta = -0.5 * angle(sum(s**2)).

Reference: Rodriguez, Calhoun & Adali (2012), Pattern Recognition 45:2050-2063.
"""

import numpy as np
from scipy.stats import skew


def correct_phase(S, A, mask=None):
    """Rotate each component onto the real axis; apply the inverse to A.

    S: (N, V) complex sources. A: (M, N) complex mixing. Returns (S, A, theta).
    A @ S is preserved exactly.
    """
    S = np.array(S, dtype=np.complex128, copy=True)
    A = np.array(A, dtype=np.complex128, copy=True)
    N, V = S.shape
    if mask is None:
        mask = np.ones(V, dtype=bool)
    mask = np.asarray(mask, dtype=bool).ravel()

    theta = np.zeros(N, dtype=np.float64)
    for k in range(N):
        sm = S[k, mask]
        th = -0.5 * np.angle(np.sum(sm**2))     # orient the major axis to the real axis
        sk = S[k] * np.exp(1j * th)
        # residual pi ambiguity: fix the direction by real-part skewness
        if skew(sk[mask].real) < 0:
            th += np.pi
            sk = -sk
        S[k] = sk
        A[:, k] = A[:, k] * np.exp(-1j * th)
        theta[k] = th
    return S, A, theta


def align_to_reference(S, S_ref):
    """Align each subject component to a reference (e.g. the group/aggregate map).

    Per-subject phase corrections are independent, so without this the residual
    rotations reintroduce non-physiological variance before group statistics.
    Returns (S_aligned, theta).
    """
    S = np.array(S, dtype=np.complex128, copy=True)
    S_ref = np.asarray(S_ref, dtype=np.complex128)
    N = S.shape[0]
    theta = np.zeros(N, dtype=np.float64)
    for k in range(N):
        th = -np.angle(np.sum(np.conj(S_ref[k]) * S[k]))
        S[k] = S[k] * np.exp(1j * th)
        theta[k] = th
    return S, theta
```

- [ ] **Step 4: Run to verify it passes**

Run: `micromamba run -n giftenv pytest python/tests/test_phase_correct.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add python/complex_gift/phase_correct.py python/tests/test_phase_correct.py
git commit -m "feat(py): phase-ambiguity correction + group phase alignment"
```

---

## Task 6: Estimator interface + noncircular complex FastICA (oracle-validated)

**Files:**
- Create: `python/complex_gift/estimators/__init__.py`
- Create: `python/complex_gift/estimators/base.py`
- Create: `python/complex_gift/estimators/nc_fastica.py`
- Create: `python/tests/test_nc_fastica.py`
- Source to translate: `GroupICAT/icatb/icatb_analysis_functions/icatb_algorithms/complex_ica/nonCircComplexFastICAsym.m` (96 lines)

**Interfaces:**
- Produces:
  - `base.py`: `class Estimator(Protocol)` with `fit(X) -> EstimatorResult`; `@dataclass EstimatorResult` with fields `W` `(N, N)`, `A` `(N, N)`, `S` `(N, T)`.
  - `nc_fastica.py`: `nc_fastica(X, nonlinearity="log", tol=1e-5, max_iter=50) -> EstimatorResult`. `nonlinearity` in `{"log", "kurt", "sqrt"}`.
  - `estimators/__init__.py`: `ESTIMATORS: dict[str, Callable]` mapping `"nc-fastica"` (and later `"cebm"`) to the callables; `get_estimator(name)`.

**Determinism note:** the reference `nonCircComplexFastICAsym.m` contains **no** random calls — it is fully deterministic. Its whitening uses `eig`, whose eigenvector order/phase differs between MATLAB and NumPy, so the recovered components may come out in a different permutation/phase than the MATLAB oracle. Compare **up to permutation and phase**, never elementwise.

- [ ] **Step 1: Write the failing oracle test**

`python/tests/test_nc_fastica.py`:

```python
import numpy as np
from scipy.io import loadmat

from complex_gift.estimators import get_estimator
from complex_gift.estimators.nc_fastica import nc_fastica
from oracle import isi, match_sources


def test_registry_exposes_nc_fastica():
    assert get_estimator("nc-fastica") is nc_fastica


def test_separates_noncircular_supergaussian_sources():
    rng = np.random.default_rng(11)
    N, T = 4, 5000
    # non-Gaussian AND noncircular. (Gaussian sources are NOT separable by ICA -
    # any rotation of whitened Gaussian data is equally valid.)
    re = rng.standard_normal((N, T)) * np.abs(rng.standard_normal((N, T))) ** 1.5
    im = rng.standard_normal((N, T)) * np.abs(rng.standard_normal((N, T))) ** 1.5
    S = re + 1j * 0.3 * im
    Amix = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    X = Amix @ S

    res = nc_fastica(X, nonlinearity="log")
    assert res.W.shape == (N, N)
    assert res.S.shape == (N, T)
    assert isi(res.W @ Amix) < 0.05


def test_matches_matlab_oracle_on_the_shared_fixture(fixtures_dir, sources):
    """Oracle diff: the port must separate the fixture at least as well as MATLAB did.

    Both are compared against the KNOWN true mixing A via ISI. (We cannot compare W
    elementwise: MATLAB's eig and NumPy's eigh order/phase eigenvectors differently, so
    the two agree only up to permutation and phase - exactly what ISI is invariant to.)
    """
    cX, A_true, cS = sources["cX"], sources["A"], sources["cS"]
    ref = loadmat(fixtures_dir / "oracle_ncfastica.mat")

    isi_matlab = isi(ref["W"] @ A_true)
    res = nc_fastica(cX, nonlinearity="log")
    isi_python = isi(res.W @ A_true)

    assert isi_python < 0.05, f"port separates poorly: ISI={isi_python:.4f}"
    assert isi_python < 2 * isi_matlab + 0.01, (
        f"port is materially worse than the MATLAB reference: "
        f"python={isi_python:.4f} matlab={isi_matlab:.4f}"
    )
    # and it recovers the true sources
    perm, corr = match_sources(res.S, cS)
    assert len(set(perm)) == cS.shape[0]      # a genuine permutation, no collisions
    assert np.all(corr > 0.9)
```

- [ ] **Step 2: Run to verify it fails**

Run: `micromamba run -n giftenv pytest python/tests/test_nc_fastica.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'complex_gift.estimators'`.

- [ ] **Step 3: Implement the interface**

`python/complex_gift/estimators/base.py`:

```python
"""Estimator interface. Every complex ICA estimator returns the same triple."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class EstimatorResult:
    W: np.ndarray   # (N, N) demixing:  S = W @ X
    A: np.ndarray   # (N, N) mixing:    X ~ A @ S
    S: np.ndarray   # (N, T) sources


class Estimator(Protocol):
    def __call__(self, X: np.ndarray, **kwargs) -> EstimatorResult: ...
```

`python/complex_gift/estimators/__init__.py`:

```python
from .base import Estimator, EstimatorResult
from .nc_fastica import nc_fastica

ESTIMATORS = {
    "nc-fastica": nc_fastica,
}


def get_estimator(name):
    """Look up an estimator by name (e.g. 'nc-fastica', 'cebm')."""
    try:
        return ESTIMATORS[name]
    except KeyError:
        raise ValueError(
            f"unknown estimator {name!r}; available: {sorted(ESTIMATORS)}"
        ) from None


__all__ = ["ESTIMATORS", "Estimator", "EstimatorResult", "get_estimator", "nc_fastica"]
```

- [ ] **Step 4: Port the estimator**

Create `python/complex_gift/estimators/nc_fastica.py` as a faithful translation of
`nonCircComplexFastICAsym.m` (96 lines). Requirements:

- GPL v3 header naming the source file and citing Novey & Adali (2008), *On extending
  the complex FastICA algorithm to noncircular sources*, IEEE TSP 56(5):2148-2154.
- Signature `nc_fastica(X, nonlinearity="log", tol=1e-5, max_iter=50) -> EstimatorResult`,
  with the reference's constants (`tol=1e-5`, `a2=0.05`, `maxcounter=50`).
- Keep the reference's structure: whiten via `eig` of the covariance, form the
  pseudo-covariance `pC = (x @ x.T) / m`, run the deflationary fixed-point updates for
  each of the three nonlinearities (`log`, `kurt`, `sqrt`), then symmetric
  orthogonalization.
- Return `W` such that `S = W @ X` (the reference returns `Ahat`; set `W = pinv(Ahat)`),
  plus `A = pinv(W)` and `S = W @ X`.
- **Translation hazards** (see Global Constraints): `'` is conjugate transpose, `.'` is
  plain transpose — the covariance uses the conjugate transpose while the
  **pseudo-covariance uses the plain transpose**; getting these backwards silently
  destroys the noncircularity the algorithm exists to exploit.

*(This step is a translation of an existing file rather than new code, so the plan
specifies the contract, the constants, and the hazards; the oracle test in Step 1 is the
executable specification of correctness.)*

- [ ] **Step 5: Run to verify it passes**

Run: `micromamba run -n giftenv pytest python/tests/test_nc_fastica.py -q`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add python/complex_gift/estimators/ python/tests/test_nc_fastica.py
git commit -m "feat(py): estimator interface + noncircular complex FastICA (oracle-validated)"
```

---

## Task 7: Complex ICA-EBM (oracle-validated) — the heaviest task

**Files:**
- Create: `python/complex_gift/estimators/cebm.py`
- Modify: `python/complex_gift/estimators/__init__.py` (register `"cebm"`)
- Create: `python/tests/test_cebm.py`
- Source to translate: `GroupICAT/icatb/icatb_analysis_functions/icatb_algorithms/complex_ica/complex_ICA_EBM.m` (802 lines)

**Interfaces:**
- Consumes: `complex_gift.nf_table.load_nf_table`, `simplified_ppval` (Task 2); `EstimatorResult` (Task 6).
- Produces: `cebm(X, nf_table_path=None, rng=None, tol=1e-4, max_iter=None) -> EstimatorResult`, registered as `"cebm"` in `ESTIMATORS`.

**Structure of the source** (translate these, in order):
| MATLAB | Lines | Notes |
|---|---|---|
| `pre_processing(X)` | 733-746 | remove DC, then whiten by `inv_sqrtmH(X @ X^H / T)` |
| `inv_sqrtmH(B)` | 747-754 | `eig` of a Hermitian matrix; `V @ diag(1/sqrt(d)) @ V^H` |
| `CEBM(X)` (main + "sea" init) | 1-268 | random init at :56, then the SEA initial-guess loop |
| `complex_ICA_EBM_north(...)` | 269-732 | the main optimizer |
| `simplified_ppval` | 755-802 | **already ported in Task 2 — import it, do not duplicate** |

**Stochasticity (important — it dictates the test design):** CEBM is **not** deterministic. It uses `randn` for the initial `W` (line 56) and `randn`/`rand` inside the optimizer (lines 342, 347, 541, 583, 634). The Python port therefore **cannot** reproduce MATLAB's `W` elementwise, and any test that tries is wrong. Instead:
- Give the port an explicit `rng` parameter (`numpy.random.Generator`, default `np.random.default_rng()`) so *its own* runs are reproducible. This is a deliberate, documented improvement over MATLAB's global RNG state.
- Validate by **separation quality against the known ground truth**, benchmarked against the MATLAB reference's separation quality on the same data.

- [ ] **Step 1: Write the failing oracle test**

`python/tests/test_cebm.py`:

```python
import numpy as np
from scipy.io import loadmat

from complex_gift.estimators import get_estimator
from complex_gift.estimators.cebm import cebm
from oracle import isi, match_sources


def test_registry_exposes_cebm():
    assert get_estimator("cebm") is cebm


def test_is_reproducible_given_a_seeded_rng(fixtures_dir, sources):
    """CEBM is stochastic; an explicit rng must make a run repeatable."""
    cX = sources["cX"]
    r1 = cebm(cX, nf_table_path=fixtures_dir / "nf_table.mat",
              rng=np.random.default_rng(0))
    r2 = cebm(cX, nf_table_path=fixtures_dir / "nf_table.mat",
              rng=np.random.default_rng(0))
    assert np.allclose(r1.W, r2.W)


def test_separates_supergaussian_sources():
    rng = np.random.default_rng(12)
    N, T = 4, 5000
    S = (rng.standard_normal((N, T)) * np.abs(rng.standard_normal((N, T))) ** 1.5
         + 1j * rng.standard_normal((N, T)) * np.abs(rng.standard_normal((N, T))) ** 1.5)
    Amix = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    X = Amix @ S
    res = cebm(X, rng=np.random.default_rng(0))
    assert isi(res.W @ Amix) < 0.05


def test_matches_matlab_oracle_on_the_shared_fixture(fixtures_dir, sources):
    """Oracle diff. CEBM is stochastic (random init + stochastic search), so MATLAB's W
    is NOT reproducible elementwise. The meaningful question is whether the port
    separates the fixture as well as the reference implementation does - measured
    against the KNOWN true mixing A.
    """
    cX, A_true, cS = sources["cX"], sources["A"], sources["cS"]
    ref = loadmat(fixtures_dir / "oracle_cebm.mat")

    isi_matlab = isi(ref["W"] @ A_true)
    res = cebm(cX, nf_table_path=fixtures_dir / "nf_table.mat",
               rng=np.random.default_rng(0))
    isi_python = isi(res.W @ A_true)

    assert isi_python < 0.05, f"port separates poorly: ISI={isi_python:.4f}"
    assert isi_python < 2 * isi_matlab + 0.01, (
        f"port is materially worse than the MATLAB reference: "
        f"python={isi_python:.4f} matlab={isi_matlab:.4f}"
    )
    perm, corr = match_sources(res.S, cS)
    assert len(set(perm)) == cS.shape[0]
    assert np.all(corr > 0.9)
```

- [ ] **Step 2: Run to verify it fails**

Run: `micromamba run -n giftenv pytest python/tests/test_cebm.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'complex_gift.estimators.cebm'`.

- [ ] **Step 3: Port the helpers first (they are short and exactly specifiable)**

In `python/complex_gift/estimators/cebm.py`, start with:

```python
def _inv_sqrtm_h(B):
    """Inverse matrix square root of a Hermitian matrix (MATLAB inv_sqrtmH)."""
    d, V = np.linalg.eigh(B)
    return (V * (1.0 / np.sqrt(d))[None, :]) @ V.conj().T


def _pre_processing(X):
    """MATLAB pre_processing: remove DC, then spatial pre-whitening. Returns (Xc, P)."""
    X = np.asarray(X, dtype=np.complex128)
    N, T = X.shape
    X = X - X.mean(axis=1, keepdims=True)
    R = X @ X.conj().T / T          # conjugate transpose: this is the covariance
    P = _inv_sqrtm_h(R)
    return P @ X, P
```

- [ ] **Step 4: Port the optimizer**

Translate `CEBM` (the SEA initial-guess loop) and `complex_ICA_EBM_north` faithfully.
Requirements:
- GPL v3 header naming the source file and citing Li & Adali (2010), *Complex independent
  component analysis by entropy bound minimization*, IEEE Trans. Circuits Syst. I,
  57(7):1417-1430.
- Import `simplified_ppval` and `load_nf_table` from `complex_gift.nf_table`; **do not
  reimplement them**. `nf_table_path` defaults to the repo's
  `complex_ica_fixtures/nf_table.mat`.
- Every MATLAB `randn`/`rand` becomes a draw from the injected `rng`
  (`rng.standard_normal(...)`, `rng.random(...)`).
- Return `EstimatorResult(W=W, A=np.linalg.pinv(W), S=W @ X)` where `W` maps the
  **original** `X` to sources (i.e. fold the pre-processing whitener `P` back in:
  the reference's returned `W` already accounts for it — check `CEBM`'s final lines and
  match its convention exactly).
- **Translation hazards** (see Global Constraints), especially: `'` (conjugate) vs `.'`
  (plain) transpose — line 59 forms the **pseudo**-covariance `C = Xc*Xc.'/T` with the
  *plain* transpose, while `pre_processing` forms the covariance with the *conjugate*
  transpose. Mixing these up is the most likely silent failure.

*(As in Task 6, this step translates an existing 800-line numerical file; the plan pins
the contract, the structure, the stochasticity handling, and the hazards, and the oracle
test in Step 1 is the executable specification of correctness.)*

- [ ] **Step 5: Register the estimator**

In `python/complex_gift/estimators/__init__.py`, add the import and registry entry:

```python
from .cebm import cebm
from .nc_fastica import nc_fastica

ESTIMATORS = {
    "cebm": cebm,
    "nc-fastica": nc_fastica,
}
```
(and add `"cebm"` to `__all__`).

- [ ] **Step 6: Run to verify it passes**

Run: `micromamba run -n giftenv pytest python/tests/test_cebm.py -q`
Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
git add python/complex_gift/estimators/cebm.py python/complex_gift/estimators/__init__.py python/tests/test_cebm.py
git commit -m "feat(py): Complex ICA-EBM port (oracle-validated)"
```

---

## Task 8: Complex NIfTI I/O

**Files:**
- Create: `python/complex_gift/complex_io.py`
- Create: `python/tests/test_complex_io.py`

**Interfaces:**
- Produces:
  - `read_complex(first_path, second_path, complex_type="real&imaginary") -> np.ndarray` — returns a complex array. `complex_type` in `{"real&imaginary", "magnitude&phase"}`. Mirrors `icatb_loadData.m:79-85`: real&imag → `first + 1j*second`; mag&phase → `first*cos(second) + 1j*first*sin(second)`.
  - `write_complex(data, first_path, second_path, affine, complex_type="real&imaginary") -> None` — splits a complex array back into two NIfTI files.
  - `complex_file_pair(path, naming=("R_", "I_")) -> (Path, Path)` — GIFT's naming convention: a prefix around an underscore (defaults `R_`/`I_`; `Mag_`/`Phase_` for magnitude&phase).

- [ ] **Step 1: Write the failing test**

`python/tests/test_complex_io.py`:

```python
import nibabel as nib
import numpy as np
import pytest

from complex_gift.complex_io import complex_file_pair, read_complex, write_complex


def _write(path, arr):
    nib.save(nib.Nifti1Image(arr.astype(np.float64), np.eye(4)), path)


def test_read_real_and_imaginary(tmp_path):
    rng = np.random.default_rng(8)
    re = rng.standard_normal((4, 4, 2))
    im = rng.standard_normal((4, 4, 2))
    _write(tmp_path / "R_sub01.nii", re)
    _write(tmp_path / "I_sub01.nii", im)
    z = read_complex(tmp_path / "R_sub01.nii", tmp_path / "I_sub01.nii",
                     complex_type="real&imaginary")
    assert z.dtype == np.complex128
    assert np.allclose(z.real, re) and np.allclose(z.imag, im)


def test_read_magnitude_and_phase(tmp_path):
    rng = np.random.default_rng(9)
    mag = np.abs(rng.standard_normal((3, 3, 2))) + 0.5
    pha = rng.uniform(-np.pi, np.pi, (3, 3, 2))
    _write(tmp_path / "Mag_sub01.nii", mag)
    _write(tmp_path / "Phase_sub01.nii", pha)
    z = read_complex(tmp_path / "Mag_sub01.nii", tmp_path / "Phase_sub01.nii",
                     complex_type="magnitude&phase")
    assert np.allclose(np.abs(z), mag)
    assert np.allclose(np.angle(z), pha)


def test_write_read_roundtrip(tmp_path):
    rng = np.random.default_rng(10)
    z = rng.standard_normal((5, 5, 3)) + 1j * rng.standard_normal((5, 5, 3))
    f1, f2 = tmp_path / "R_out.nii", tmp_path / "I_out.nii"
    write_complex(z, f1, f2, affine=np.eye(4), complex_type="real&imaginary")
    z2 = read_complex(f1, f2, complex_type="real&imaginary")
    assert np.allclose(z, z2)


def test_file_pair_naming():
    a, b = complex_file_pair("/data/sub01_run1.nii", naming=("R_", "I_"))
    assert a.name == "R_sub01_run1.nii"
    assert b.name == "I_sub01_run1.nii"


def test_file_pair_requires_underscore():
    """GIFT's naming convention keys on an underscore; be explicit rather than silent."""
    with pytest.raises(ValueError, match="underscore"):
        complex_file_pair("/data/sub01.nii", naming=("R_", "I_"))
```

- [ ] **Step 2: Run to verify it fails**

Run: `micromamba run -n giftenv pytest python/tests/test_complex_io.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'complex_gift.complex_io'`.

- [ ] **Step 3: Implement**

`python/complex_gift/complex_io.py`:

```python
"""Complex NIfTI I/O.

GIFT stores complex data as TWO ordinary NIfTI files per volume (image-domain
reconstructed data, not raw k-space), distinguished by a prefix around an underscore:
R_/I_ for real&imaginary, Mag_/Phase_ for magnitude&phase.
Mirrors icatb_loadData.m:79-85.
"""

from pathlib import Path

import nibabel as nib
import numpy as np

REAL_IMAG = "real&imaginary"
MAG_PHASE = "magnitude&phase"


def read_complex(first_path, second_path, complex_type=REAL_IMAG):
    """Assemble one complex array from GIFT's two-file representation."""
    a = np.asarray(nib.load(str(first_path)).get_fdata(), dtype=np.float64)
    b = np.asarray(nib.load(str(second_path)).get_fdata(), dtype=np.float64)
    if complex_type == REAL_IMAG:
        return (a + 1j * b).astype(np.complex128)
    if complex_type == MAG_PHASE:
        return (a * np.cos(b) + 1j * a * np.sin(b)).astype(np.complex128)
    raise ValueError(f"unknown complex_type {complex_type!r}")


def write_complex(data, first_path, second_path, affine, complex_type=REAL_IMAG):
    """Split a complex array back into two NIfTI files."""
    data = np.asarray(data, dtype=np.complex128)
    if complex_type == REAL_IMAG:
        a, b = data.real, data.imag
    elif complex_type == MAG_PHASE:
        a, b = np.abs(data), np.angle(data)
    else:
        raise ValueError(f"unknown complex_type {complex_type!r}")
    nib.save(nib.Nifti1Image(a, affine), str(first_path))
    nib.save(nib.Nifti1Image(b, affine), str(second_path))


def complex_file_pair(path, naming=("R_", "I_")):
    """Derive GIFT's two filenames from a base name (prefix around an underscore)."""
    p = Path(path)
    if "_" not in p.name:
        raise ValueError(
            f"complex file names must contain an underscore (GIFT convention): {p.name}"
        )
    return p.with_name(naming[0] + p.name), p.with_name(naming[1] + p.name)
```

- [ ] **Step 4: Run to verify it passes**

Run: `micromamba run -n giftenv pytest python/tests/test_complex_io.py -q`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add python/complex_gift/complex_io.py python/tests/test_complex_io.py
git commit -m "feat(py): complex NIfTI I/O (R_/I_ and Mag_/Phase_)"
```

---

## Task 9: Group pipeline + end-to-end integration test

**Files:**
- Create: `python/complex_gift/group.py`
- Create: `python/complex_gift/pipeline.py`
- Create: `python/tests/test_pipeline.py`

**Interfaces:**
- Consumes: every unit from Tasks 2-8.
- Produces:
  - `group.py`:
    - `two_stage_pca(subject_data, n_subject, n_group) -> (Xg, reduced, whiteners, W_group)`.
      `subject_data` is a list of `(T_i, V)` complex arrays. `Xg` is `(n_group, V)`;
      `reduced` is a list of per-subject `(n_subject, V)` arrays; `whiteners` is a list of
      per-subject `(n_subject, T_i)` matrices; `W_group` is `(n_group, n_sub*n_subject)`.
    - `back_reconstruct(S_group, A_group, reduced, whiteners, W_group) -> list[(S_i, A_i)]`
      — GICA back-reconstruction. `S_i` is `(N, V)`, `A_i` is `(T_i, N)`.
  - `pipeline.py`: `run_complex_ica(subject_data, n_components, estimator="cebm", mag_mask=None, rng=None) -> PipelineResult` — the whole thing: phase mask → two-stage PCA → estimator → phase correction → back-reconstruction → per-subject group phase alignment. `@dataclass PipelineResult` with `S_group`, `A_group`, `mask`, `subjects: list[tuple[S_i, A_i]]`.

- [ ] **Step 1: Write the failing end-to-end test**

`python/tests/test_pipeline.py`:

```python
import numpy as np

from complex_gift.pipeline import run_complex_ica
from oracle import match_sources


def _subject(rng, Smaps, phi, T):
    """One subject's complex fMRI-like data, (T, V).

    Physical model: z(v,t) = rho(v,t) * exp(i*phi(v)) - a POSITIVE magnitude series
    (static baseline + BOLD-like modulation) times a STATIC per-voxel background phase.
    That is what makes signal voxels phase-stable over time (what the mask keys on) while
    keeping each component's voxels clustered along a line (what phase correction needs).
    """
    N, Vsig = Smaps.shape
    tc = rng.standard_normal((T, N))
    base = 100.0 + 10.0 * rng.random(Vsig)
    M = base[None, :] + 5.0 * (tc @ Smaps)      # positive magnitude
    assert np.all(M > 0)
    return M * np.exp(1j * phi)[None, :]


def test_end_to_end_group_complex_ica():
    rng = np.random.default_rng(21)
    N, Vsig, Vnoise, T, n_sub = 3, 1000, 300, 60, 4
    V = Vsig + Vnoise

    Smaps = rng.standard_normal((N, Vsig)) * np.abs(rng.standard_normal((N, Vsig))) ** 1.5
    Smaps /= np.abs(Smaps).max()
    phi = (rng.random(Vsig) * 20 - 10) / 180 * np.pi      # static phase, +/-10 deg

    subjects = []
    for _ in range(n_sub):
        sig = _subject(rng, Smaps, phi, T)                             # (T, Vsig)
        noise = (0.5 + rng.random((T, Vnoise))) * np.exp(
            1j * 2 * np.pi * rng.random((T, Vnoise))
        )
        subjects.append(np.concatenate([sig, noise], axis=1))          # (T, V)

    res = run_complex_ica(subjects, n_components=N, estimator="cebm",
                          rng=np.random.default_rng(0))

    # the phase mask kept the signal voxels and dropped the random-phase ones
    assert res.mask[:Vsig].mean() > 0.9
    assert res.mask[Vsig:].mean() < 0.05

    # the group decomposition recovered the true maps (up to permutation/phase)
    Strue = (Smaps * np.exp(1j * phi)[None, :])[:, res.mask[:Vsig]]
    perm, corr = match_sources(res.S_group, Strue)
    assert len(set(perm)) == N
    assert np.all(corr > 0.8)

    # phase correction left the maps concentrated on the real axis
    imag_frac = (res.S_group.imag**2).sum(axis=1) / (np.abs(res.S_group) ** 2).sum(axis=1)
    assert np.all(imag_frac < 0.15)

    # back-reconstruction produced one map set per subject, aligned to the group
    assert len(res.subjects) == n_sub
    for S_i, A_i in res.subjects:
        assert S_i.shape == res.S_group.shape
        p_i, c_i = match_sources(S_i, res.S_group)
        assert list(p_i) == list(range(N))    # already aligned: identity permutation
        assert np.all(c_i > 0.8)
```

- [ ] **Step 2: Run to verify it fails**

Run: `micromamba run -n giftenv pytest python/tests/test_pipeline.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'complex_gift.pipeline'`.

- [ ] **Step 3: Implement the group unit**

`python/complex_gift/group.py`:

```python
"""Group complex ICA: two-stage PCA reduction and GICA back-reconstruction.

Follows GIFT's standard scheme: reduce each subject, temporally concatenate, reduce
again at the group level, run ICA on the group-reduced data, then back-reconstruct
per-subject maps and time courses.
"""

import numpy as np

from .whiten import whiten_hermitian


def two_stage_pca(subject_data, n_subject, n_group):
    """Subject-level then group-level complex PCA (temporal concatenation).

    subject_data: list of (T_i, V) complex arrays (already masked).
    Returns (Xg (n_group, V), reduced [list of (n_subject, V)],
             whiteners [list of (n_subject, T_i)], W_group (n_group, n_sub*n_subject)).
    """
    reduced, whiteners = [], []
    for Xi in subject_data:
        Xi = np.asarray(Xi, dtype=np.complex128)                    # (T_i, V)
        Yi, W_i, _ = whiten_hermitian(Xi, n_components=n_subject)   # (n_subject, V)
        reduced.append(Yi)
        whiteners.append(W_i)

    stacked = np.concatenate(reduced, axis=0)             # (n_sub*n_subject, V)
    Xg, W_group, _ = whiten_hermitian(stacked, n_components=n_group)
    return Xg, reduced, whiteners, W_group


def back_reconstruct(S_group, A_group, reduced, whiteners, W_group):
    """GICA back-reconstruction to subject-specific maps and time courses.

    The group model is Xg = A_group @ S_group in the group-reduced space. Undo the group
    whitener to express the group mixing in the STACKED subject-reduced space:

        B = pinv(W_group) @ A_group          # (n_sub*n_subject, N)

    and partition B by subject into Bi (n_subject, N). Bi maps the sources into subject
    i's reduced space, so subject i's own maps come from projecting that subject's ACTUAL
    reduced data through it:

        S_i = pinv(Bi) @ Y_i                 # (N, V)

    Using Y_i (the real data) rather than Bi @ S_group is the whole point - the latter
    would collapse to S_group for every subject and carry no subject-specific variance.

    Returns a list of (S_i (N, V), A_i (T_i, N)).
    """
    n_sub = len(reduced)
    B = np.linalg.pinv(W_group) @ A_group                 # (n_sub*n_subject, N)
    blocks = np.split(B, n_sub, axis=0)                   # each (n_subject, N)

    out = []
    for Y_i, W_i, Bi in zip(reduced, whiteners, blocks):
        S_i = np.linalg.pinv(Bi) @ Y_i                    # subject-specific maps (N, V)
        A_i = np.linalg.pinv(W_i) @ Bi                    # subject time courses (T_i, N)
        out.append((S_i, A_i))
    return out
```

- [ ] **Step 4: Implement the pipeline**

`python/complex_gift/pipeline.py`:

```python
"""The complete complex ICA pipeline, wired end to end."""

from dataclasses import dataclass

import numpy as np

from .estimators import get_estimator
from .group import back_reconstruct, two_stage_pca
from .phase_correct import align_to_reference, correct_phase
from .phase_mask import phase_quality_mask


@dataclass
class PipelineResult:
    S_group: np.ndarray             # (N, V_masked) group maps
    A_group: np.ndarray             # group mixing
    mask: np.ndarray                # (V,) boolean
    subjects: list                  # [(S_i, A_i), ...] back-reconstructed, group-aligned


def run_complex_ica(subject_data, n_components, estimator="cebm", mag_mask=None,
                    n_subject=None, rng=None, **estimator_kwargs):
    """Run group complex ICA over a list of (T_i, V) complex subject arrays."""
    if rng is None:
        rng = np.random.default_rng()
    if n_subject is None:
        n_subject = n_components

    # 1. phase-quality mask, computed on the concatenated complex data (V, T_total)
    Z = np.concatenate([np.asarray(x, dtype=np.complex128) for x in subject_data],
                       axis=0).T                                   # (V, T_total)
    mask, _, _ = phase_quality_mask(Z, mag_mask=mag_mask)

    masked = [np.asarray(x, dtype=np.complex128)[:, mask] for x in subject_data]

    # 2. two-stage complex PCA reduction
    Xg, reduced, whiteners, W_group = two_stage_pca(masked, n_subject, n_components)

    # 3. complex ICA
    est = get_estimator(estimator)
    res = est(Xg, rng=rng, **estimator_kwargs) if estimator == "cebm" \
        else est(Xg, **estimator_kwargs)

    # 4. phase-ambiguity correction on the group maps
    S_group, A_group, _ = correct_phase(res.S, res.A)

    # 5. back-reconstruct per subject, then align each to the group reference
    subjects = []
    for S_i, A_i in back_reconstruct(S_group, A_group, reduced, whiteners, W_group):
        S_i, _ = align_to_reference(S_i, S_group)
        subjects.append((S_i, A_i))

    return PipelineResult(S_group=S_group, A_group=A_group, mask=mask, subjects=subjects)
```

- [ ] **Step 5: Run to verify it passes**

Run: `micromamba run -n giftenv pytest python/tests/test_pipeline.py -q`
Expected: 1 passed.

- [ ] **Step 6: Run the whole suite**

Run: `micromamba run -n giftenv pytest python/tests -q`
Expected: all tests pass (Tasks 1-9).

- [ ] **Step 7: Commit**

```bash
git add python/complex_gift/group.py python/complex_gift/pipeline.py python/tests/test_pipeline.py
git commit -m "feat(py): group two-stage PCA, back-reconstruction, end-to-end pipeline"
```

> **Phase 2 exit criterion (spec §9):** Python matches the MATLAB oracle within tolerance
> (Tasks 6, 7); the phase-step property tests pass (Tasks 4, 5); an end-to-end group run
> recovers known sources (Task 9). Deliverable: the `complex-gift` package plus
> `complex_ica_fixtures/nf_table.npz`, the canonical lookup-table export that the Rust
> track (Phase 3) consumes.

---

## Notes for the implementer

- **Everything runs in `giftenv`:** `micromamba run -n giftenv pytest python/tests -q`. Run pytest from the `python/` directory (or rely on `pyproject.toml`'s `testpaths`).
- **ICA cannot separate Gaussian sources.** Every test that asserts source recovery must use non-Gaussian (super-Gaussian) sources. This is an identifiability limit, not a tuning problem — it already cost a debugging cycle in the MATLAB phase. If a recovery test fails, check the source distribution *first*.
- **Never compare estimator weight matrices elementwise.** CEBM is stochastic and both estimators are only defined up to permutation and per-source complex phase. Compare with `isi()` and `match_sources()`.
- **Tasks 6 and 7 are translations**, not fresh code: the plan pins the contract, structure, and hazards, and the oracle tests are the executable specification. Read the MATLAB source completely before writing Python — partial understanding of a numerical algorithm guarantees subtle bugs.
- **The fixtures are read-only.** If a test seems to want different fixture data, the test is wrong.
