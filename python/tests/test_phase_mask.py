import numpy as np

from complex_gift.phase_mask import otsu_threshold, phase_quality_mask, quality_map


def _synthetic(rng, n_stable=250, n_random=250, T=80):
    """Stable-phase voxels (signal) and random-phase voxels (noise)."""
    V = n_stable + n_random
    mag = 1.0 + 0.1 * rng.standard_normal((V, T))
    phi = np.empty((V, T))
    phi[:n_stable] = 0.1 * rng.standard_normal((n_stable, T))     # stable
    phi[n_stable:] = 2 * np.pi * rng.random((n_random, T))        # random
    return mag * np.exp(1j * phi), n_stable


def test_quality_map_separates_stable_from_random_phase():
    rng = np.random.default_rng(3)
    Z, n = _synthetic(rng)
    Q = quality_map(Z)
    assert np.all((Q >= 0) & (Q <= 1))
    assert Q[:n].mean() > Q[n:].mean() + 0.3


def test_mask_selects_stable_voxels():
    rng = np.random.default_rng(3)
    Z, n = _synthetic(rng)
    mask, Q, tau = phase_quality_mask(Z)
    assert mask[:n].mean() > 0.8
    assert mask[n:].mean() < 0.2


def test_invariance_to_constant_per_voxel_phase():
    """THE defining property: a static per-voxel phase offset (B0/receiver phase)
    must not change Q or the mask at all."""
    rng = np.random.default_rng(3)
    Z, _ = _synthetic(rng)
    offset = np.exp(1j * 2 * np.pi * rng.random((Z.shape[0], 1)))  # constant over time
    mask1, Q1, _ = phase_quality_mask(Z)
    mask2, Q2, _ = phase_quality_mask(Z * offset)
    assert np.abs(Q1 - Q2).max() < 1e-10
    assert np.array_equal(mask1, mask2)


def test_otsu_splits_a_bimodal_distribution():
    rng = np.random.default_rng(4)
    x = np.concatenate([rng.normal(0.1, 0.02, 500), rng.normal(0.9, 0.02, 500)])
    tau = otsu_threshold(x)
    assert 0.2 < tau < 0.8
