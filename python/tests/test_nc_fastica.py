import numpy as np
from scipy.io import loadmat

from complex_gift.estimators import get_estimator
from complex_gift.estimators.nc_fastica import nc_fastica
from oracle import isi, match_sources


def test_registry_exposes_nc_fastica():
    assert get_estimator("nc-fastica") is nc_fastica


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

    res = nc_fastica(X, nonlinearity="log")
    assert res.W.shape == (N, N)
    assert res.S.shape == (N, T)
    assert isi(res.W @ Amix) < 0.05


def test_matches_matlab_oracle_on_the_shared_fixture(fixtures_dir, sources):
    """Oracle diff: the port must separate the fixture at least as well as MATLAB did.

    Both are compared against the KNOWN true mixing A via ISI. (We cannot compare W
    elementwise: MATLAB's eig and NumPy's eigh order/phase eigenvectors differently, so
    the two agree only up to permutation and phase - exactly what ISI is invariant to.)
    """
    cX, A_true, cS = sources["cX"], sources["A"], sources["cS"]
    ref = loadmat(fixtures_dir / "oracle_ncfastica.mat")

    isi_matlab = isi(ref["W"] @ A_true)
    res = nc_fastica(cX, nonlinearity="log")
    isi_python = isi(res.W @ A_true)

    assert isi_python < 0.05, f"port separates poorly: ISI={isi_python:.4f}"
    assert isi_python < 2 * isi_matlab + 0.01, (
        f"port is materially worse than the MATLAB reference: "
        f"python={isi_python:.4f} matlab={isi_matlab:.4f}"
    )
    # and it recovers the true sources
    perm, corr = match_sources(res.S, cS)
    assert len(set(perm)) == cS.shape[0]      # a genuine permutation, no collisions
    assert np.all(corr > 0.9)


def test_default_max_iter_matches_reference_cap():
    """Verify default max_iter=None reproduces MATLAB's effective cap of 15*n.

    The MATLAB reference has a dead local maxcounter=50, but its actual loop
    bound is 15*n. The default must use 15*n, not 50. This test verifies that
    max_iter=None and max_iter=15*n produce identical results for N=4
    (where 15*4=60 != 50, catching any regression to the old default).
    """
    rng = np.random.default_rng(42)
    N, T = 4, 1000  # 4 components: 15*4=60 != 50
    re = rng.standard_normal((N, T)) * np.abs(rng.standard_normal((N, T))) ** 1.5
    im = rng.standard_normal((N, T)) * np.abs(rng.standard_normal((N, T))) ** 1.5
    X = re + 1j * 0.3 * im
    Amix = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    X = Amix @ X

    # Call with max_iter=None (default, should use 15*4=60 internally)
    res_default = nc_fastica(X, nonlinearity="log", max_iter=None)

    # Call with explicit max_iter=15*N (should produce identical results)
    res_explicit = nc_fastica(X, nonlinearity="log", max_iter=15 * N)

    # Both must produce identical W. This verifies that max_iter=None correctly
    # uses 15*n. If someone reverts to max_iter=50, this test would fail
    # because the explicit 15*N call would allow more iterations than 50.
    assert np.allclose(res_default.W, res_explicit.W, atol=1e-12), (
        "max_iter=None and max_iter=15*n should produce identical W"
    )
