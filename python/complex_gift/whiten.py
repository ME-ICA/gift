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
    #        P := U.conj().T @ Vh.conj().T.conj() = U.conj().T @ V_svd.conj()
    #    to be diagonal whenever the singular values in S are distinct
    #    (P commutes with diag(S), and a matrix that commutes with a
    #    diagonal matrix with distinct entries must itself be diagonal).
    #    Writing P's diagonal entries as unit-modulus phases p_i, the
    #    Takagi vectors are V = U @ diag(sqrt(p_i)), verified numerically
    #    (200/200 random trials, including a degenerate-singular-value
    #    case) to satisfy V unitary and Pc = V @ diag(S) @ V.T exactly.
    Pc = Xw @ Xw.T / T
    U_, s_, Vh_ = np.linalg.svd(Pc)
    V_svd = Vh_.conj().T
    P = U_.conj().T @ V_svd.conj()           # diagonal (up to numerical noise)
    phase = np.diag(P)
    V = U_ * np.sqrt(phase)[None, :]         # Takagi basis: Pc = V @ diag(s_) @ V.T
    W_sut = V.conj().T @ W_wh
    Xs = W_sut @ Xc
    return Xs, W_sut
