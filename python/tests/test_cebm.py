import numpy as np
from scipy.io import loadmat

from complex_gift.estimators import get_estimator
from complex_gift.estimators.cebm import cebm
from oracle import isi, match_sources


def test_registry_exposes_cebm():
    assert get_estimator("cebm") is cebm


def test_is_reproducible_given_a_seeded_rng(fixtures_dir, sources):
    """CEBM is stochastic; an explicit rng must make a run repeatable."""
    cX = sources["cX"]
    r1 = cebm(cX, nf_table_path=fixtures_dir / "nf_table.mat",
              rng=np.random.default_rng(0))
    r2 = cebm(cX, nf_table_path=fixtures_dir / "nf_table.mat",
              rng=np.random.default_rng(0))
    assert np.allclose(r1.W, r2.W)


def test_separates_supergaussian_sources():
    rng = np.random.default_rng(12)
    N, T = 4, 5000
    S = (rng.standard_normal((N, T)) * np.abs(rng.standard_normal((N, T))) ** 1.5
         + 1j * rng.standard_normal((N, T)) * np.abs(rng.standard_normal((N, T))) ** 1.5)
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
    cX, A_true, cS = sources["cX"], sources["A"], sources["cS"]
    ref = loadmat(fixtures_dir / "oracle_cebm.mat")

    isi_matlab = isi(ref["W"] @ A_true)
    res = cebm(cX, nf_table_path=fixtures_dir / "nf_table.mat",
               rng=np.random.default_rng(0))
    isi_python = isi(res.W @ A_true)

    assert isi_python < 0.05, f"port separates poorly: ISI={isi_python:.4f}"
    assert isi_python < 2 * isi_matlab + 0.01, (
        f"port is materially worse than the MATLAB reference: "
        f"python={isi_python:.4f} matlab={isi_matlab:.4f}"
    )
    perm, corr = match_sources(res.S, cS)
    assert len(set(perm)) == cS.shape[0]
    assert np.all(corr > 0.9)
