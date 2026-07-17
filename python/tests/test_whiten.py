import numpy as np
import pytest

from gift.whiten import strong_uncorrelating_transform, whiten_hermitian


def test_hermitian_whitening_decorrelates():
    rng = np.random.default_rng(1)
    P, T, N = 8, 4000, 4
    S = rng.standard_normal((N, T)) + 1j * rng.standard_normal((N, T))
    Amix = rng.standard_normal((P, N)) + 1j * rng.standard_normal((P, N))
    X = Amix @ S
    Xw, _W_wh, _W_dw = whiten_hermitian(X, n_components=N)
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
    Xw, _W_wh, W_dw = whiten_hermitian(X, n_components=N)
    # data is exactly rank N, so dewhitening must reconstruct it
    assert np.allclose(W_dw @ Xw, Xc, atol=1e-6)


def test_raises_when_more_components_than_rows():
    rng = np.random.default_rng(4)
    P, T = 5, 500
    X = rng.standard_normal((P, T)) + 1j * rng.standard_normal((P, T))
    with pytest.raises(ValueError, match='exceeds the number of'):
        whiten_hermitian(X, n_components=12)


def test_raises_on_rank_deficient_input():
    # Build 6 rows that are exact linear combinations of 3 independent complex rows,
    # so the true rank of X is 3 regardless of T. Requesting more components than the
    # rank must raise rather than silently emit NaNs from sqrt of a ~0 eigenvalue.
    rng = np.random.default_rng(5)
    T = 500
    base = rng.standard_normal((3, T)) + 1j * rng.standard_normal((3, T))
    mix = rng.standard_normal((6, 3)) + 1j * rng.standard_normal((6, 3))
    X = mix @ base  # (6, T), exactly rank 3
    with pytest.raises(ValueError, match='rank-deficient'):
        whiten_hermitian(X, n_components=4)


def test_sut_diagonalizes_covariance_and_pseudo_covariance():
    rng = np.random.default_rng(3)
    N, T = 4, 20000
    # noncircular sources: unequal real/imag variance => nonzero pseudo-covariance
    S = rng.standard_normal((N, T)) + 1j * 0.3 * rng.standard_normal((N, T))
    Amix = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    X = Amix @ S
    Xs, _W_sut = strong_uncorrelating_transform(X)
    C = Xs @ Xs.conj().T / T  # covariance -> identity
    Pc = Xs @ Xs.T / T  # pseudo-covariance -> diagonal (real, nonneg)
    assert np.allclose(C, np.eye(N), atol=1e-2)
    off = Pc - np.diag(np.diag(Pc))
    assert np.abs(off).max() < 1e-2 * max(1.0, np.abs(np.diag(Pc)).max())


def test_sut_handles_degenerate_singular_values():
    # Build data whose whitened pseudo-covariance has *exactly* repeated
    # singular values, by algebraic construction rather than statistical
    # cancellation (random circular sources are only circular in
    # expectation; their finite-sample pseudo-covariance has generic,
    # non-degenerate singular values and does not reproduce the bug).
    #
    # For each of N components i, place a value a_i at time index i and
    # -a_i at time index N+i (all other entries zero). This makes each row
    # exactly zero-mean, gives an exactly isotropic Hermitian covariance
    # Xc @ Xc^H / T = c * I, and an exactly diagonal pseudo-covariance
    # Xc @ Xc^T / T = diag(a_i^2 ...) whose diagonal entries all have the
    # same magnitude c (an N-fold exactly-degenerate singular value), by
    # picking |a_i|^2 constant and only varying each a_i's phase.
    #
    # That data alone is already diagonal in this basis, so it does *not*
    # expose the bug (SVD of an already-diagonal matrix is trivially
    # diagonal too). Applying a random *unitary* mixing Q rotates the
    # pseudo-covariance to Q @ diag(...) @ Q.T -- still exactly N-fold
    # degenerate in its singular values, but no longer diagonal -- which is
    # exactly the regime where the naive "take diag(U^H @ V_svd.conj())"
    # Takagi shortcut picks entries that are not unit-modulus phases and
    # returns a measurably non-unitary V. Confirmed against the buggy
    # implementation: this construction gave invariant errors of ~0.4
    # (vs. the tolerances of 1e-2 asserted below).
    rng = np.random.default_rng(7)
    N = 4
    c = 1.0
    theta = 0.37 * np.arange(1, N + 1)
    a = np.sqrt(c / 2) * np.exp(1j * theta)

    Xbase = np.zeros((N, 2 * N), dtype=np.complex128)
    for i in range(N):
        Xbase[i, i] = a[i]
        Xbase[i, N + i] = -a[i]

    Araw = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    Q, _ = np.linalg.qr(Araw)
    X = Q @ Xbase
    T = X.shape[1]

    Xs, _W_sut = strong_uncorrelating_transform(X)
    C = Xs @ Xs.conj().T / T  # covariance -> identity
    Pc = Xs @ Xs.T / T  # pseudo-covariance -> diagonal (real, nonneg)
    assert np.allclose(C, np.eye(N), atol=1e-2)
    off = Pc - np.diag(np.diag(Pc))
    assert np.abs(off).max() < 1e-2 * max(1.0, np.abs(np.diag(Pc)).max())
