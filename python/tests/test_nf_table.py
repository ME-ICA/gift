import numpy as np
import pytest
from scipy.interpolate import PPoly

from gift.nf_table import export_nf_table, load_nf_table, simplified_ppval


@pytest.fixture(scope='module')
def nf(fixtures_dir):
    return load_nf_table(fixtures_dir / 'nf_table.mat')


def test_structure(nf):
    assert sorted(nf) == [f'nf{i}' for i in range(1, 9)]
    for v in nf.values():
        assert v.pp.coefs.shape == (v.pp.pieces, 4)
        assert v.pp.breaks.shape == (v.pp.pieces + 1,)
        assert v.pp.order == 4
        assert v.max_EGx > v.min_EGx


def test_ppval_matches_scipy_ppoly(nf):
    """Cross-check the hand-ported evaluator against an independent implementation.

    MATLAB pp: on [breaks[i], breaks[i+1]) the value is
        sum_j coefs[i, j] * (x - breaks[i])**(order-1-j)
    which is exactly scipy PPoly with c = coefs.T. Two independent evaluators
    agreeing is strong evidence the port of simplified_ppval is faithful.
    """
    for v in nf.values():
        for pp in (v.pp, v.pp_slope):
            ref = PPoly(c=pp.coefs.T.copy(), x=pp.breaks.copy())
            xs = np.linspace(pp.breaks[0], pp.breaks[-1], 97)[1:-1]
            got = np.array([simplified_ppval(pp, float(x)) for x in xs])
            assert np.allclose(got, ref(xs), rtol=1e-10, atol=1e-12)


def test_ppval_clamps_outside_breaks(nf):
    """Outside the knot range MATLAB's evaluator extrapolates from the end piece
    (index clamped to first/last), rather than raising."""
    pp = nf['nf1'].pp
    assert np.isfinite(simplified_ppval(pp, float(pp.breaks[0] - 5.0)))
    assert np.isfinite(simplified_ppval(pp, float(pp.breaks[-1] + 5.0)))


def test_export_npz_roundtrip(nf, fixtures_dir, tmp_path):
    out = tmp_path / 'nf_table.npz'
    export_nf_table(fixtures_dir / 'nf_table.mat', out)
    d = np.load(out)
    for name, v in nf.items():
        assert np.array_equal(d[f'{name}/pp/breaks'], v.pp.breaks)
        assert np.array_equal(d[f'{name}/pp/coefs'], v.pp.coefs)
        assert np.array_equal(d[f'{name}/pp_slope/coefs'], v.pp_slope.coefs)
        assert float(d[f'{name}/min_EGx']) == v.min_EGx
        # Handle NaN comparison: if both are NaN, they match
        cp2_exported = float(d[f'{name}/critical_point2'])
        if np.isnan(v.critical_point2):
            assert np.isnan(cp2_exported), f'{name} critical_point2 mismatch'
        else:
            assert cp2_exported == v.critical_point2
