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

    Note: `critical_point2` is present only on `nf1` in the source table and is
    `NaN` for `nf2`–`nf8` by design of the source MATLAB struct. This field is
    unused by the algorithm and is inert legacy metadata.
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
            critical_point2=float(getattr(m, 'critical_point2', np.nan)),
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
