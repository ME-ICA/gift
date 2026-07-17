"""Nonlinearity lookup tables for Complex ICA-EBM.

GPL v3 - derives from the Adali-lab (MLSP/UMBC) complex ICA-EBM implementation.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.io import loadmat

_NF_NAMES = [f'nf{i}' for i in range(1, 9)]


@dataclass(frozen=True)
class PP:
    """MATLAB piecewise-polynomial (spline) form.

    Attributes
    ----------
    breaks : numpy.ndarray of float64, shape (pieces + 1,)
        The knots delimiting the polynomial pieces.
    coefs : numpy.ndarray of float64, shape (pieces, order)
        Per-piece polynomial coefficients in descending powers of the local coordinate
        ``x - breaks[i]``.
    pieces : int
        The number of polynomial pieces.
    order : int
        The polynomial order, i.e. the number of coefficients per piece.
    """

    breaks: np.ndarray
    coefs: np.ndarray
    pieces: int
    order: int


@dataclass(frozen=True)
class NF:
    """One entropy-bound nonlinearity: its scalars, its spline, and the spline's slope.

    Attributes
    ----------
    min_EGx, max_EGx : float
        The range of ``E[G(x)]`` over which the spline is valid. Outside it, callers
        extrapolate linearly from the endpoint slope.
    critical_point : float
        Threshold used by the algorithm to select among nonlinearities.
    critical_point2 : float
        Inert legacy metadata. Present only on ``nf1`` in the source table and ``NaN``
        for ``nf2`` through ``nf8`` by design of the source MATLAB struct. Unused by the
        algorithm, and retained only so the export matches the source table field for
        field.
    pp : PP
        The spline giving the entropy bound as a function of ``E[G(x)]``.
    pp_slope : PP
        The spline giving that function's slope.
    """

    min_EGx: float
    max_EGx: float
    critical_point: float
    critical_point2: float
    pp: PP
    pp_slope: PP


def _to_pp(struct) -> PP:
    """Convert one loaded MATLAB piecewise-polynomial struct into a :class:`PP`.

    Parameters
    ----------
    struct : scipy.io.matlab.mat_struct
        A MATLAB ``pp`` struct, as returned by :func:`scipy.io.loadmat` with
        ``struct_as_record=False``.

    Returns
    -------
    PP
        The converted piecewise-polynomial form.
    """
    return PP(
        breaks=np.asarray(struct.breaks, dtype=np.float64).ravel(),
        coefs=np.asarray(struct.coefs, dtype=np.float64),
        pieces=int(struct.pieces),
        order=int(struct.order),
    )


def load_nf_table(path) -> dict[str, NF]:
    """Load ``nf_table.mat`` or ``complex_nf_table.mat`` into plain Python objects.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the MATLAB table file.

    Returns
    -------
    dict of {str: NF}
        The eight nonlinearities, keyed ``'nf1'`` through ``'nf8'``.

    Notes
    -----
    The tables ship as MATLAB structs holding piecewise-polynomial (spline) forms, which
    are unpacked here into dataclasses so the rest of the package never handles
    scipy's MATLAB struct proxies.

    References
    ----------
    .. [1] Li, X.-L., & Adali, T. (2010). Complex independent component analysis by
           entropy bound minimization. IEEE Transactions on Circuits and Systems I,
           57(7), 1417-1430.
    """
    table = loadmat(path, squeeze_me=True, struct_as_record=False)
    out = {}
    for name in _NF_NAMES:
        struct = table[name]
        out[name] = NF(
            min_EGx=float(struct.min_EGx),
            max_EGx=float(struct.max_EGx),
            critical_point=float(struct.critical_point),
            critical_point2=float(getattr(struct, 'critical_point2', np.nan)),
            pp=_to_pp(struct.pp),
            pp_slope=_to_pp(struct.pp_slope),
        )
    return out


def simplified_ppval(pp: PP, x: float) -> float:
    """Evaluate a piecewise polynomial at one point.

    Parameters
    ----------
    pp : PP
        The piecewise-polynomial form to evaluate.
    x : float
        The point to evaluate at. Points outside ``pp.breaks`` are clamped to the first
        or last piece, whose polynomial is then extrapolated.

    Returns
    -------
    float
        The polynomial's value at ``x``.

    Notes
    -----
    This is an exact port of the evaluator bundled with the reference implementation
    (``complex_ICA_EBM.m`` lines 755-802). It is ported rather than replaced with a
    library spline so that the numerics match the reference exactly.

    Evaluation has three steps: binary-search for the piece containing ``x``, shift to
    that piece's local coordinate ``x - breaks[i]``, then evaluate by nested (Horner)
    multiplication.

    Two details of the port are load-bearing. The reference's ``round((low + high) / 2)``
    uses MATLAB rounding, which is half away from zero, whereas Python's :func:`round`
    is banker's rounding; ``floor(x + 0.5)`` reproduces MATLAB exactly. And the piece
    index is computed in MATLAB's 1-based convention, matching the reference line for
    line, then converted to 0-based only when indexing.

    References
    ----------
    .. [1] Li, X.-L., & Adali, T. (2010). Complex independent component analysis by
           entropy bound minimization. IEEE Transactions on Circuits and Systems I,
           57(7), 1417-1430.
    """
    breaks = pp.breaks
    coefs = pp.coefs
    n_pieces = pp.pieces  # MATLAB `l`
    order = 4  # the reference hardcodes order 4

    # Find the piece index, in MATLAB's 1-based convention.
    if x > breaks[n_pieces - 1]:  # MATLAB: xs > b(l)
        piece = n_pieces  # MATLAB: index = l
    elif x < breaks[1]:  # MATLAB: xs < b(2)
        piece = 1  # MATLAB: index = 1
    else:
        low_index = 1
        high_index = n_pieces
        while True:
            middle_index = math.floor((low_index + high_index) / 2 + 0.5)
            if breaks[middle_index - 1] > x:
                high_index = middle_index
            else:
                low_index = middle_index
            if low_index == high_index - 1:
                piece = low_index
                break

    piece_index = piece - 1  # to 0-based row of coefs / breaks

    # Shift to local coordinates, then evaluate by nested multiplication.
    local_x = x - breaks[piece_index]
    value = coefs[piece_index, 0]
    for power in range(1, order):
        value = local_x * value + coefs[piece_index, power]
    return float(value)


def export_nf_table(mat_path, npz_path) -> None:
    """Write the language-neutral canonical export consumed by the Rust port.

    Parameters
    ----------
    mat_path : str or pathlib.Path
        Path to the source MATLAB table file.
    npz_path : str or pathlib.Path
        Path for the ``.npz`` output.

    Notes
    -----
    The MATLAB struct hierarchy is flattened into slash-separated keys (for example
    ``nf1/pp/breaks``), because ``.npz`` holds a flat name-to-array mapping. Reading the
    export therefore needs no MATLAB, and no struct traversal.
    """
    nf_table = load_nf_table(mat_path)
    flat = {}
    for name, nf in nf_table.items():
        flat[f'{name}/min_EGx'] = np.float64(nf.min_EGx)
        flat[f'{name}/max_EGx'] = np.float64(nf.max_EGx)
        flat[f'{name}/critical_point'] = np.float64(nf.critical_point)
        flat[f'{name}/critical_point2'] = np.float64(nf.critical_point2)
        for attr in ('pp', 'pp_slope'):
            pp = getattr(nf, attr)
            flat[f'{name}/{attr}/breaks'] = pp.breaks
            flat[f'{name}/{attr}/coefs'] = pp.coefs
    np.savez(npz_path, **flat)
