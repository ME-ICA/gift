"""Complex whitening.

`whiten_hermitian` mirrors GIFT's icatb_pca_whitening.m: it uses only the
covariance E[xx^H], which is the right thing for circular (proper) sources.

`strong_uncorrelating_transform` additionally diagonalizes the pseudo-covariance
E[xx^T], which carries signal when sources are noncircular - required by
noncircular estimators such as nc-FastICA.
"""

import numpy as np


def whiten_hermitian(X, n_components=None):
    """PCA-whiten complex data using the Hermitian covariance.

    X: (P, T) complex. Returns (Xw (N, T), W_whiten (N, P), W_dewhiten (P, N)).
    """
    X = np.asarray(X, dtype=np.complex128)
    P, T = X.shape
    N = P if n_components is None else int(n_components)

    Xc = X - X.mean(axis=1, keepdims=True)
    R = Xc @ Xc.conj().T / T                # (P, P) Hermitian; note conjugate transpose
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

    # 2. the whitened pseudo-covariance Pc = Xw @ Xw.T / T is complex
    #    SYMMETRIC (not Hermitian). Its Takagi factorization
    #    Pc = V @ diag(k) @ V.T, with V unitary and k >= 0 real, gives the
    #    unitary rotation Q = V.conj().T that simultaneously keeps
    #    Q @ Pc_cov @ Q.conj().T = I (any unitary preserves the whitened
    #    covariance) and diagonalizes Pc via Q @ Pc @ Q.T = diag(k).
    #
    #    Derivation of the Takagi basis from an ordinary SVD Pc = U S Vh:
    #    since Pc is symmetric, Pc = Pc.T forces
    #        Z := U.conj().T @ Vh.conj().T.conj() = U.conj().T @ V_svd.conj()
    #    to commute with diag(S). Z is unitary (product of two unitaries).
    #    When the singular values in S are all distinct, that commutation
    #    forces Z to be diagonal with unit-modulus entries p_i, and the
    #    Takagi vectors are V = U @ diag(sqrt(p_i)).
    #
    #    That shortcut breaks down under degeneracy: whenever S has
    #    repeated (or several exactly zero) singular values, Z is only
    #    forced to be *block-diagonal* within each degenerate group, not
    #    diagonal -- taking diag(Z) then silently picks non-unit-modulus
    #    entries, and the resulting V is measurably non-unitary (confirmed
    #    bug: this happens for realistic fMRI data with multiple perfectly
    #    circular components). The general, degeneracy-safe fix is to use
    #    the unitary *square root* of Z rather than its diagonal: for
    #    unitary Z, eigendecomposition Z = Q @ diag(w) @ Q^-1 has |w| == 1,
    #    and sqrtm(Z) = Q @ diag(sqrt(w)) @ Q^-1 is well-defined for any
    #    eigenvalue multiplicity (the degenerate-eigenspace projector is
    #    unaffected by which eigenbasis Q we get within that eigenspace).
    #    V = U @ sqrtm(Z) then satisfies V unitary and
    #    Pc = V @ diag(S) @ V.T exactly, in both the generic and the
    #    degenerate case (verified numerically, see test_whiten.py's
    #    test_sut_handles_degenerate_singular_values).
    Pc = Xw @ Xw.T / T
    U_, s_, Vh_ = np.linalg.svd(Pc)
    V_svd = Vh_.conj().T
    Z = U_.conj().T @ V_svd.conj()           # unitary; diagonal only if S has no repeats
    w, Qz = np.linalg.eig(Z)
    w = w / np.abs(w)                        # guard against roundoff drift off |w| == 1
    sqrtZ = Qz @ np.diag(np.sqrt(w)) @ np.linalg.inv(Qz)
    V = U_ @ sqrtZ                            # Takagi basis: Pc = V @ diag(s_) @ V.T

    N_ = V.shape[0]
    unitarity_err = np.abs(V.conj().T @ V - np.eye(N_)).max()
    if not np.isfinite(unitarity_err) or unitarity_err > 1e-6:
        raise RuntimeError(
            "strong_uncorrelating_transform: Takagi factor V failed the "
            f"unitarity check (max |V^H V - I| = {unitarity_err:.3g} > 1e-6); "
            "refusing to return a silently wrong transform."
        )

    W_sut = V.conj().T @ W_wh
    Xs = W_sut @ Xc
    return Xs, W_sut
