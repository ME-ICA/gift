import numpy as np
from oracle import isi, match_sources
from scipy.io import loadmat

from gift.estimators import get_estimator
from gift.estimators.cebm import _pre_processing, _pseudo_cov, cebm


def test_registry_exposes_cebm():
    assert get_estimator('cebm') is cebm


def test_pseudo_cov_uses_plain_transpose():
    """The pseudo-covariance E[x x^T] must use the PLAIN transpose, not the
    conjugate transpose. Swapping to `.conj().T` silently destroys the
    noncircularity CEBM exploits while still looking numerically plausible
    (the existing ISI-based oracle test cannot catch this regression).
    """
    rng = np.random.default_rng(3)
    N, T = 4, 2000
    # Genuinely noncircular complex data: real and imaginary parts correlated.
    real = rng.standard_normal((N, T))
    imag = 0.7 * real + 0.3 * rng.standard_normal((N, T))
    X = real + 1j * imag

    plain = X @ X.T / T
    hermitian = X @ X.conj().T / T

    result = _pseudo_cov(X)

    assert np.allclose(result, plain)
    assert not np.allclose(result, hermitian)


def test_pre_processing_whitens_with_hermitian_covariance(sources):
    """`_pre_processing` must whiten using the HERMITIAN (conjugate-transpose)
    covariance, not the pseudo-covariance. The fixture data is genuinely
    noncircular, so pinning both conventions on the same output distinguishes
    them unambiguously.
    """
    cX = sources['cX']
    T = cX.shape[1]

    Xc, _ = _pre_processing(cX)

    cov = Xc @ Xc.conj().T / T
    pseudo_cov = Xc @ Xc.T / T

    assert np.allclose(cov, np.eye(cov.shape[0]), atol=1e-8)
    assert not np.allclose(pseudo_cov, np.eye(pseudo_cov.shape[0]), atol=0.1)


def test_is_reproducible_given_a_seeded_rng(fixtures_dir, sources):
    """CEBM is stochastic; an explicit rng must make a run repeatable."""
    cX = sources['cX']
    r1 = cebm(cX, nf_table_path=fixtures_dir / 'nf_table.mat', rng=np.random.default_rng(0))
    r2 = cebm(cX, nf_table_path=fixtures_dir / 'nf_table.mat', rng=np.random.default_rng(0))
    assert np.allclose(r1.W, r2.W)


def test_separates_supergaussian_sources():
    rng = np.random.default_rng(12)
    N, T = 4, 5000
    S = (
        rng.standard_normal((N, T)) * np.abs(rng.standard_normal((N, T))) ** 1.5
        + 1j * rng.standard_normal((N, T)) * np.abs(rng.standard_normal((N, T))) ** 1.5
    )
    Amix = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    X = Amix @ S
    res = cebm(X, rng=np.random.default_rng(0))
    assert isi(res.W @ Amix) < 0.05


def test_separates_at_larger_n_incremental_branch():
    """N=10 (>7) exercises the incremental Sherman-Morrison `inv_Q` update path in
    `_north`, which the N=6/N=4 fixtures used elsewhere never reach."""
    rng = np.random.default_rng(7)
    N, T = 10, 5000
    S = (
        rng.standard_normal((N, T)) * np.abs(rng.standard_normal((N, T))) ** 1.5
        + 1j * rng.standard_normal((N, T)) * np.abs(rng.standard_normal((N, T))) ** 1.5
    )
    Amix = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    X = Amix @ S
    res = cebm(X, rng=np.random.default_rng(0))
    assert isi(res.W @ Amix) < 0.05


def test_matches_matlab_oracle_on_the_shared_fixture(fixtures_dir, sources):
    """Oracle diff. CEBM is stochastic (random init + stochastic search), so MATLAB's W
    is NOT reproducible elementwise. The meaningful question is whether the port
    separates the fixture as well as the reference implementation does - measured
    against the KNOWN true mixing A.
    """
    cX, A_true, cS = sources['cX'], sources['A'], sources['cS']
    ref = loadmat(fixtures_dir / 'oracle_cebm.mat')

    isi_matlab = isi(ref['W'] @ A_true)
    res = cebm(cX, nf_table_path=fixtures_dir / 'nf_table.mat', rng=np.random.default_rng(0))
    isi_python = isi(res.W @ A_true)

    assert isi_python < 0.05, f'port separates poorly: ISI={isi_python:.4f}'
    assert isi_python < 2 * isi_matlab + 0.01, (
        f'port is materially worse than the MATLAB reference: '
        f'python={isi_python:.4f} matlab={isi_matlab:.4f}'
    )
    perm, corr = match_sources(res.S, cS)
    assert len(set(perm)) == cS.shape[0]
    assert np.all(corr > 0.9)
