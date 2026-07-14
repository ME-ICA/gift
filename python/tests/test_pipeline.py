import numpy as np

from complex_gift.pipeline import run_complex_ica
from oracle import match_sources


def _subject(rng, Smaps, phi, T):
    """One subject's complex fMRI-like data, (T, V).

    Physical model: z(v,t) = rho(v,t) * exp(i*phi(v)) - a POSITIVE magnitude series
    (static baseline + BOLD-like modulation) times a STATIC per-voxel background phase.
    That is what makes signal voxels phase-stable over time (what the mask keys on) while
    keeping each component's voxels clustered along a line (what phase correction needs).
    """
    N, Vsig = Smaps.shape
    tc = rng.standard_normal((T, N))
    base = 100.0 + 10.0 * rng.random(Vsig)
    M = base[None, :] + 5.0 * (tc @ Smaps)      # positive magnitude
    assert np.all(M > 0)
    return M * np.exp(1j * phi)[None, :]


def test_end_to_end_group_complex_ica():
    rng = np.random.default_rng(21)
    N, Vsig, Vnoise, T, n_sub = 3, 1000, 300, 60, 4
    V = Vsig + Vnoise

    Smaps = rng.standard_normal((N, Vsig)) * np.abs(rng.standard_normal((N, Vsig))) ** 1.5
    Smaps /= np.abs(Smaps).max()
    phi = (rng.random(Vsig) * 20 - 10) / 180 * np.pi      # static phase, +/-10 deg

    subjects = []
    for _ in range(n_sub):
        sig = _subject(rng, Smaps, phi, T)                             # (T, Vsig)
        noise = (0.5 + rng.random((T, Vnoise))) * np.exp(
            1j * 2 * np.pi * rng.random((T, Vnoise))
        )
        subjects.append(np.concatenate([sig, noise], axis=1))          # (T, V)

    res = run_complex_ica(subjects, n_components=N, estimator="cebm",
                          rng=np.random.default_rng(0))

    # the phase mask kept the signal voxels and dropped the random-phase ones
    assert res.mask[:Vsig].mean() > 0.9
    assert res.mask[Vsig:].mean() < 0.05

    # The group decomposition recovered the true maps (up to permutation/phase).
    # Build the ground truth over ALL voxels (noise voxels carry no signal, hence zeros)
    # and then apply the SAME mask the pipeline used. The mask is allowed to leak a few
    # noise voxels (the assertion above only bounds leakage at <5%), so indexing the
    # signal block alone would produce a shape mismatch the moment one leaks through.
    StrueFull = np.concatenate(
        [Smaps * np.exp(1j * phi)[None, :], np.zeros((N, Vnoise), dtype=np.complex128)],
        axis=1,
    )                                                        # (N, V)
    Strue = StrueFull[:, res.mask]                           # (N, V_masked)
    perm, corr = match_sources(res.S_group, Strue)
    assert len(set(perm)) == N
    assert np.all(corr > 0.8)

    # phase correction left the maps concentrated on the real axis
    imag_frac = (res.S_group.imag**2).sum(axis=1) / (np.abs(res.S_group) ** 2).sum(axis=1)
    assert np.all(imag_frac < 0.15)

    # back-reconstruction produced one map set per subject, aligned to the group
    assert len(res.subjects) == n_sub
    for S_i, A_i in res.subjects:
        assert S_i.shape == res.S_group.shape
        p_i, c_i = match_sources(S_i, res.S_group)
        assert list(p_i) == list(range(N))    # already aligned: identity permutation
        assert np.all(c_i > 0.8)
