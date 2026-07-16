"""Noncircular complex FastICA (symmetric orthogonalization).

Ported from GroupICAT/icatb/icatb_analysis_functions/icatb_algorithms/complex_ica/
nonCircComplexFastICAsym.m.

Reference:
    Mike Novey and T. Adali, "On Extending the complex FastICA algorithm to
    noncircular sources," IEEE Trans. Signal Processing, 56(5):2148-2154, May 2008.

Copyright (C) 2023 MLSP Lab (original MATLAB); this Python translation.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
details: <https://www.gnu.org/licenses/>.
"""

import numpy as np

from .base import EstimatorResult
from .cebm import _pseudo_cov

_A2 = 0.05  # nonlinearity smoothing constant (reference: a2)
_NONLINEARITIES = ("log", "kurt", "sqrt")


def _g_gp(nonlinearity, absy):
    """Nonlinearity g and its "derivative-like" companion gp, per the reference."""
    if nonlinearity == "log":
        g = 1.0 / (_A2 + absy)
        gp = -1.0 / (_A2 + absy) ** 2
    elif nonlinearity == "kurt":
        g = absy
        gp = np.ones_like(absy)
    elif nonlinearity == "sqrt":
        g = 1.0 / (2.0 * np.sqrt(_A2 + absy))
        gp = -1.0 / (4.0 * (_A2 + absy) ** 1.5)
    else:
        raise ValueError(
            f"unknown nonlinearity {nonlinearity!r}; expected one of {_NONLINEARITIES}"
        )
    return g, gp


def resolve_max_iter(max_iter, n):
    """Resolve the iteration cap.

    MATLAB's ``maxcounter = 50`` is DEAD CODE; the reference's real loop bound is
    ``15 * n`` (nonCircComplexFastICAsym.m line 60). ``None`` therefore resolves to
    ``15 * n``, NOT 50.

    Split out so the default is directly assertable: realistic fixtures converge long
    before either cap bites, so comparing OUTPUTS cannot tell 50 from 15*n.
    """
    return 15 * n if max_iter is None else max_iter


def nc_fastica(X, nonlinearity="log", tol=1e-5, max_iter=None):
    """Noncircular complex FastICA, symmetric orthogonalization.

    X: (N, T) complex mixtures. Fully deterministic (the reference has no
    rand/randn calls). Returns an EstimatorResult with S = W @ X.

    max_iter: maximum iterations. If None (default), uses the reference's
    effective cap of 15*n (where n is the number of components). The MATLAB
    reference assigns maxcounter=50 but never uses it; the actual loop bound
    is 15*n. Passing an explicit integer overrides this behavior.
    """
    if nonlinearity not in _NONLINEARITIES:
        raise ValueError(
            f"unknown nonlinearity {nonlinearity!r}; expected one of {_NONLINEARITIES}"
        )

    xold = np.asarray(X, dtype=np.complex128)
    n, m = xold.shape

    max_iter = resolve_max_iter(max_iter, n)

    # Whitening: eig(cov(xold')) in MATLAB. MATLAB's cov() computes the
    # CONJUGATE (Hermitian) covariance with N-1 normalization; np.cov matches
    # this exactly when xold's rows are the variables (its default).
    cov = np.cov(xold)
    Dx, Ex = np.linalg.eigh(cov)  # ascending real eigenvalues, orthonormal eigenvectors
    Q = (np.sqrt(1.0 / Dx))[:, None] * Ex.conj().T  # sqrt(inv(Dx)) @ Ex'
    x = Q @ xold

    # Pseudo-covariance: PLAIN transpose (NOT conjugate) - this is what lets the
    # algorithm exploit noncircularity. Do not change `.T` to `.conj().T` here.
    # Pseudo-covariance E[x x^T]: PLAIN transpose (NOT conjugate). Share CEBM's named,
    # separately-guarded helper rather than inlining it -- a code comment is not a guard.
    # A whole-phase review found that corrupting this inline to `.conj().T` left ALL 41
    # tests green (ISI merely doubles, well inside the bound), which is the exact bug class
    # a Phase-2 review PROVED the ISI oracle cannot catch.
    pC = _pseudo_cov(x)

    W = np.eye(n, dtype=np.complex128)
    Wold = np.zeros((n, n), dtype=np.complex128)
    k = 0

    # Convergence condition: loop while orthonormality deviation exceeds threshold
    # AND iteration count is within the limit (max_iter, which defaults to 15*n).
    while (
        np.linalg.norm(np.abs(Wold.conj().T @ W) - np.eye(n), "fro") > (n * tol)
        and k < max_iter
    ):
        k += 1
        Wold = W.copy()

        for kk in range(n):
            wold_col = Wold[:, kk]
            yy = wold_col.conj() @ x  # W(:,kk)' * x
            absy = np.abs(yy) ** 2

            g, gp = _g_gp(nonlinearity, absy)

            gRad = np.mean(x * (g * yy.conj())[None, :], axis=1)
            ggg = np.mean(gp * absy + g)
            B = np.mean(gp * yy.conj() ** 2) * pC

            W[:, kk] = wold_col * ggg - gRad + B @ wold_col.conj()

        # Symmetric orthonormalization: W = W * E * inv(sqrt(D)) * E'
        D, E = np.linalg.eigh(W.conj().T @ W)
        W = W @ E @ np.diag(1.0 / np.sqrt(D)) @ E.conj().T

    Ahat = np.linalg.inv(Q) @ W

    W_out = np.linalg.pinv(Ahat)
    A_out = np.linalg.pinv(W_out)
    S_out = W_out @ xold

    return EstimatorResult(W=W_out, A=A_out, S=S_out)
