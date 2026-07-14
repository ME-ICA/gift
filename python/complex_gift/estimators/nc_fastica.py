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


def nc_fastica(X, nonlinearity="log", tol=1e-5, max_iter=50):
    """Noncircular complex FastICA, symmetric orthogonalization.

    X: (N, T) complex mixtures. Fully deterministic (the reference has no
    rand/randn calls). Returns an EstimatorResult with S = W @ X.
    """
    if nonlinearity not in _NONLINEARITIES:
        raise ValueError(
            f"unknown nonlinearity {nonlinearity!r}; expected one of {_NONLINEARITIES}"
        )

    xold = np.asarray(X, dtype=np.complex128)
    n, m = xold.shape

    # Whitening: eig(cov(xold')) in MATLAB. MATLAB's cov() computes the
    # CONJUGATE (Hermitian) covariance with N-1 normalization; np.cov matches
    # this exactly when xold's rows are the variables (its default).
    cov = np.cov(xold)
    Dx, Ex = np.linalg.eigh(cov)  # ascending real eigenvalues, orthonormal eigenvectors
    Q = (np.sqrt(1.0 / Dx))[:, None] * Ex.conj().T  # sqrt(inv(Dx)) @ Ex'
    x = Q @ xold

    # Pseudo-covariance: PLAIN transpose (NOT conjugate) - this is what lets the
    # algorithm exploit noncircularity. Do not change `.T` to `.conj().T` here.
    pC = (x @ x.T) / m

    W = np.eye(n, dtype=np.complex128)
    Wold = np.zeros((n, n), dtype=np.complex128)
    k = 0

    # NOTE: the MATLAB reference bounds this loop with the literal `15*n`
    # (its `maxcounter=50` local is dead code, never referenced). Since
    # `max_iter` is now a real, user-facing parameter, we honor it as the
    # actual iteration cap rather than reproducing the dead-code literal.
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
