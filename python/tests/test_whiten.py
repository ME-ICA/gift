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
