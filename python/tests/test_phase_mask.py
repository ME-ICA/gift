import numpy as np
import pytest
from scipy.ndimage import uniform_filter

from gift.phase_mask import (
    combine_subject_masks,
    otsu_threshold,
    pdv_quality_map,
    phase_quality_mask,
    quality_map,
    wrap_phase,
)


def _synthetic(rng, n_stable=250, n_random=250, T=80):
    """Stable-phase voxels (signal) and random-phase voxels (noise)."""
    V = n_stable + n_random
    mag = 1.0 + 0.1 * rng.standard_normal((V, T))
    phi = np.empty((V, T))
    phi[:n_stable] = 0.1 * rng.standard_normal((n_stable, T))  # stable
    phi[n_stable:] = 2 * np.pi * rng.random((n_random, T))  # random
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
    mask, _Q, _tau = phase_quality_mask(Z)
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
    # Otsu's between-class variance is FLAT across the empty gap between two
    # well-separated modes (moving the threshold there reclassifies no points), so
    # argmax legitimately lands anywhere in the gap - pinning tau to a sub-range would
    # only test a tie-breaking detail. Assert what actually matters: it separates them.
    assert (x[:500] < tau).mean() > 0.99
    assert (x[500:] > tau).mean() > 0.99


# --- PDV (spatial phase-derivative variance), ported from Complex GIFT --------------


def _matlab_wrap(x):
    """Literal port of icatb_wrap_func.m, elementwise."""
    x = np.array(x, dtype=np.float64)
    out = x.copy()
    it = np.nditer(x, flags=['multi_index'])
    for v in it:
        val = float(v)
        if val > np.pi:
            while val > np.pi:
                val -= 2 * np.pi
        elif val <= -np.pi:
            while val <= -np.pi:
                val += 2 * np.pi
        out[it.multi_index] = val
    return out


def _matlab_val_qual_circular(phs_img, wrap_flag=1):
    """Line-for-line port of icatb_val_qual_circular.m for a 2-D slice.

    This is an INDEPENDENT reference (explicit per-pixel loops, exactly as the MATLAB
    reads) against which the vectorized pdv_quality_map is checked. Two implementations
    agreeing is strong evidence the fast one is faithful.
    """
    k = 3
    sz = phs_img.shape
    val_mat = np.zeros(sz)

    FY = np.zeros(sz)
    FY[:-1, :] = np.diff(phs_img, axis=0)
    FY[-1, :] = phs_img[0, :] - phs_img[-1, :]

    FX = np.zeros(sz)
    FX[:, :-1] = np.diff(phs_img, axis=1)
    FX[:, -1] = phs_img[:, 0] - phs_img[:, -1]

    if wrap_flag == 1:
        FX = _matlab_wrap(FX)
        FY = _matlab_wrap(FY)

    avg_FX = uniform_filter(FX, size=3, mode='wrap')
    avg_FY = uniform_filter(FY, size=3, mode='wrap')

    for i in range(sz[0]):
        for j in range(sz[1]):
            corner = k // 2
            x_sum = 0.0
            y_sum = 0.0
            for m in range(-corner, corner + 1):
                for n in range(-corner, corner + 1):
                    nr = (i + m) % sz[0]  # MATLAB wraps out-of-bounds indices circularly
                    nc = (j + n) % sz[1]
                    x_sum += (FX[nr, nc] - avg_FX[i, j]) ** 2
                    y_sum += (FY[nr, nc] - avg_FY[i, j]) ** 2
            val_mat[i, j] = (np.sqrt(x_sum) + np.sqrt(y_sum)) / k**2
    return val_mat


def test_wrap_phase_matches_reference_including_endpoints():
    rng = np.random.default_rng(0)
    x = np.concatenate(
        [rng.uniform(-4 * np.pi, 4 * np.pi, 3000), [np.pi, -np.pi, 3 * np.pi, -3 * np.pi]]
    )
    assert np.allclose(wrap_phase(x), _matlab_wrap(x), atol=1e-12)
    # the half-open interval: exactly -pi maps to +pi, unlike np.angle
    assert wrap_phase(-np.pi) == pytest.approx(np.pi)


def test_pdv_quality_map_matches_literal_transcription():
    rng = np.random.default_rng(1)
    for _ in range(10):
        h, w = int(rng.integers(6, 18)), int(rng.integers(6, 18))
        phase = wrap_phase(rng.uniform(-np.pi, np.pi, (h, w)))
        ref = _matlab_val_qual_circular(phase, wrap_flag=1)
        got = pdv_quality_map(phase, wrap=True)
        assert np.allclose(got, ref, atol=1e-10)


def test_pdv_broadcasts_over_trailing_axes():
    """A 4-D (X, Y, S, T) call must equal per-slice, per-timepoint application."""
    rng = np.random.default_rng(2)
    vol = wrap_phase(rng.uniform(-np.pi, np.pi, (12, 9, 3, 4)))
    bulk = pdv_quality_map(vol)
    for s in range(vol.shape[2]):
        for t in range(vol.shape[3]):
            assert np.allclose(pdv_quality_map(vol[:, :, s, t]), bulk[:, :, s, t], atol=1e-12)


def test_pdv_is_invariant_to_constant_phase_offset():
    """Built from phase differences, so a whole-slice offset must cancel."""
    rng = np.random.default_rng(3)
    phase = wrap_phase(rng.uniform(-np.pi, np.pi, (14, 11)))
    shifted = wrap_phase(phase + 1.234)
    assert np.allclose(pdv_quality_map(phase), pdv_quality_map(shifted), atol=1e-10)


def _pdv_synthetic(rng, dims=(40, 40, 2), T=15):
    """A mostly-smooth-phase volume with a random-phase (noise) pocket.

    PDV keys on SPATIAL phase smoothness, so the signal must be a large smooth region --
    not a small block, whose sharp boundary against the noise would itself read as high
    variance and whose interior the per-slice erosion would consume. Real brain volumes
    are large and mostly smooth, which is the regime this reproduces.
    """
    x, y, s = dims
    V = x * y * s
    noise = np.zeros((x, y, s), dtype=bool)
    noise[8:20, 8:20, :] = True  # a compact random-phase pocket
    good = (~noise).reshape(V)
    noise = noise.reshape(V)

    phase = np.empty((V, T))
    phase[good] = 0.02 * rng.standard_normal((int(good.sum()), T))  # smooth everywhere
    phase[noise] = np.pi * (2 * rng.random((int(noise.sum()), T)) - 1)  # random pocket
    Z = np.exp(1j * phase)  # unit magnitude; PDV only looks at phase
    return Z, good, noise, dims


def test_pdv_mask_keeps_the_smooth_region_and_drops_noise():
    rng = np.random.default_rng(4)
    Z, good, noise, dims = _pdv_synthetic(rng)
    mask, Q, tau = phase_quality_mask(Z, method='pdv', dims=dims)
    assert tau == 0.2  # the reference's fixed threshold, returned as-is
    assert mask[good].mean() > 0.8
    assert mask[noise].mean() < 0.05
    # the returned PDV summary map is lower (better) in the smooth region than the pocket
    assert Q[good].mean() < Q[noise].mean()


def test_pdv_requires_dims():
    rng = np.random.default_rng(5)
    Z, *_ = _pdv_synthetic(rng)
    with pytest.raises(ValueError, match='requires dims'):
        phase_quality_mask(Z, method='pdv')


def test_pdv_rejects_mismatched_dims():
    rng = np.random.default_rng(6)
    Z, *_ = _pdv_synthetic(rng)
    with pytest.raises(ValueError, match='voxels'):
        phase_quality_mask(Z, method='pdv', dims=(5, 5, 5))


def test_unknown_method_raises():
    rng = np.random.default_rng(7)
    Z, *_ = _pdv_synthetic(rng)
    with pytest.raises(ValueError, match='unknown method'):
        phase_quality_mask(Z, method='nonsense')


def test_combine_subject_masks_votes_by_agreement():
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    c = np.array([True, True, True, False])
    # agreement=0.8 over 3 subjects => need count >= 2.4 => >= 3 subjects
    kept = combine_subject_masks([a, b, c], agreement=0.8)
    assert list(kept) == [True, False, False, False]
    # agreement=0.5 => need >= 1.5 => >= 2 subjects
    kept = combine_subject_masks([a, b, c], agreement=0.5)
    assert list(kept) == [True, True, True, False]


def test_combine_subject_masks_rejects_shape_mismatch():
    with pytest.raises(ValueError, match='shape'):
        combine_subject_masks([np.zeros(4, bool), np.zeros(3, bool)])
