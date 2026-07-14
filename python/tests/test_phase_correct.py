import numpy as np

from complex_gift.phase_correct import align_to_reference, correct_phase


def _aligned_sources(rng, N=4, V=3000):
    """Sources whose energy already lies along the real axis."""
    return rng.standard_normal((N, V)) + 1j * 0.05 * rng.standard_normal((N, V))


def test_reconstruction_is_preserved():
    """The whole point: rotating S by e^{i0} and A by e^{-i0} must leave A @ S alone."""
    rng = np.random.default_rng(5)
    N, V, M = 4, 3000, 20
    S0 = _aligned_sources(rng, N, V)
    A0 = rng.standard_normal((M, N)) + 1j * rng.standard_normal((M, N))
    theta_true = np.array([0.7, -1.1, 0.3, 2.0])
    S = S0 * np.exp(1j * theta_true)[:, None]
    A = A0 * np.exp(-1j * theta_true)[None, :]

    X_before = A @ S
    Sc, Ac, theta = correct_phase(S, A)
    rel = np.linalg.norm(X_before - Ac @ Sc) / np.linalg.norm(X_before)
    assert rel < 1e-10


def test_injected_rotation_is_removed():
    """A known rotation is undone: corrected sources concentrate on the real axis."""
    rng = np.random.default_rng(5)
    N, V, M = 4, 3000, 20
    S0 = _aligned_sources(rng, N, V)
    A0 = rng.standard_normal((M, N)) + 1j * rng.standard_normal((M, N))
    theta_true = np.array([0.7, -1.1, 0.3, 2.0])
    S = S0 * np.exp(1j * theta_true)[:, None]
    A = A0 * np.exp(-1j * theta_true)[None, :]

    Sc, Ac, theta = correct_phase(S, A)
    imag_frac = (Sc.imag**2).sum(axis=1) / (np.abs(Sc) ** 2).sum(axis=1)
    assert np.all(imag_frac < 0.05)
    # residual major-axis angle is ~0 mod pi
    resid = np.mod(np.angle((Sc**2).sum(axis=1)), np.pi)
    assert np.all(np.minimum(resid, np.pi - resid) < 1e-3)


def test_mask_restricts_the_estimate():
    """theta is estimated only from masked (high-quality) voxels: garbage outside the
    mask must not move the answer."""
    rng = np.random.default_rng(6)
    N, V, M = 3, 2000, 10
    S = _aligned_sources(rng, N, V)
    A = rng.standard_normal((M, N)) + 1j * rng.standard_normal((M, N))
    mask = np.zeros(V, dtype=bool)
    mask[: V // 2] = True
    S_dirty = S.copy()
    S_dirty[:, ~mask] *= 50.0 * np.exp(1j * 1.3)      # wild values outside the mask

    _, _, theta_clean = correct_phase(S.copy(), A.copy(), mask=mask)
    _, _, theta_dirty = correct_phase(S_dirty, A.copy(), mask=mask)
    assert np.allclose(theta_clean, theta_dirty, atol=1e-9)


def test_group_alignment_collapses_known_subject_rotations():
    rng = np.random.default_rng(7)
    N, V = 3, 1500
    S_ref = _aligned_sources(rng, N, V)
    rot = np.array([0.9, -0.4, 2.2])
    S_subj = S_ref * np.exp(1j * rot)[:, None]
    S_al, theta = align_to_reference(S_subj, S_ref)
    assert np.allclose(np.abs(theta), np.abs(rot), atol=1e-6)
    # aligned sources now agree with the reference up to a positive real scale
    for k in range(N):
        c = np.vdot(S_ref[k], S_al[k]) / np.vdot(S_ref[k], S_ref[k])
        assert abs(c.imag) < 1e-6
