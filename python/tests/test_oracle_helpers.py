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
