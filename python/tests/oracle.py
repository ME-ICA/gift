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
    C = np.abs(Se @ St.conj().T)  # (n_est, n_true)
    perm = C.argmax(axis=1)
    return perm, C.max(axis=1)
