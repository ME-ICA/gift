import numpy as np
from oracle import isi, match_sources
from scipy.io import loadmat

from gift.estimators import get_estimator
from gift.estimators.nc_fastica import nc_fastica, resolve_max_iter


def test_registry_exposes_nc_fastica():
    assert get_estimator('nc-fastica') is nc_fastica


def test_separates_noncircular_supergaussian_sources():
    rng = np.random.default_rng(11)
    N, T = 4, 5000
    # non-Gaussian AND noncircular. (Gaussian sources are NOT separable by ICA -
    # any rotation of whitened Gaussian data is equally valid.)
    re = rng.standard_normal((N, T)) * np.abs(rng.standard_normal((N, T))) ** 1.5
    im = rng.standard_normal((N, T)) * np.abs(rng.standard_normal((N, T))) ** 1.5
    S = re + 1j * 0.3 * im
    Amix = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    X = Amix @ S

    res = nc_fastica(X, nonlinearity='log')
    assert res.W.shape == (N, N)
    assert res.S.shape == (N, T)
    assert isi(res.W @ Amix) < 0.05


def test_matches_matlab_oracle_on_the_shared_fixture(fixtures_dir, sources):
    """Oracle diff: the port must separate the fixture at least as well as MATLAB did.

    Both are compared against the KNOWN true mixing A via ISI. (We cannot compare W
    elementwise: MATLAB's eig and NumPy's eigh order/phase eigenvectors differently, so
    the two agree only up to permutation and phase - exactly what ISI is invariant to.)
    """
    cX, A_true, cS = sources['cX'], sources['A'], sources['cS']
    ref = loadmat(fixtures_dir / 'oracle_ncfastica.mat')

    isi_matlab = isi(ref['W'] @ A_true)
    res = nc_fastica(cX, nonlinearity='log')
    isi_python = isi(res.W @ A_true)

    assert isi_python < 0.05, f'port separates poorly: ISI={isi_python:.4f}'
    # nc-FastICA is FULLY DETERMINISTIC (the reference has no rand/randn), so this is not
    # a "close enough" comparison -- the port reproduces MATLAB's ISI to ~1e-15. The old
    # bound (2*isi_matlab + 0.01) allowed 3.4x drift, which is dishonest for a
    # deterministic estimator AND hid real bugs: corrupting the pseudo-covariance to the
    # conjugate transpose merely doubles ISI, which the loose bound absorbed silently.
    assert abs(isi_python - isi_matlab) < 1e-6, (
        f"deterministic port must reproduce MATLAB's ISI: "
        f'python={isi_python:.12f} matlab={isi_matlab:.12f}'
    )
    # and it recovers the true sources
    perm, corr = match_sources(res.S, cS)
    assert len(set(perm)) == cS.shape[0]  # a genuine permutation, no collisions
    assert np.all(corr > 0.9)


def test_default_max_iter_resolves_to_the_reference_cap():
    """The default cap must be 15*n, not MATLAB's dead maxcounter=50.

    Assert the RESOLUTION, not the outputs. Realistic fixtures converge in ~16
    iterations, so neither 50 nor 15*n ever bites and comparing W from max_iter=None
    against max_iter=15*n passes even when the default is wrong. (Verified: reverting
    the default to 50 left the old output-comparison test green.) This mirrors the
    Rust port's resolve_max_iter test.
    """
    assert resolve_max_iter(None, 6) == 90
    assert resolve_max_iter(None, 4) == 60
    assert resolve_max_iter(None, 2) == 30
    # an explicit value always wins, including one that differs from both caps
    assert resolve_max_iter(7, 6) == 7
    assert resolve_max_iter(50, 4) == 50
