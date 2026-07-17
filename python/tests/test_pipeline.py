import numpy as np
from oracle import match_sources

from gift.pipeline import run_complex_ica


def _subject(rng, Smaps, phi, T, noise=0.0):
    """One subject's complex fMRI-like data, (T, V).

    Physical model: z(v,t) = rho(v,t) * exp(i*phi(v)) - a POSITIVE magnitude series
    (static baseline + BOLD-like modulation) times a STATIC per-voxel background phase.
    That is what makes signal voxels phase-stable over time (what the mask keys on) while
    keeping each component's voxels clustered along a line (what phase correction needs).

    `noise` adds subject-specific magnitude noise. With noise=0 the de-meaned signal is
    exactly rank N, so a reduction can only ever ask for N components; real subjects carry
    variance beyond the shared sources, which is what makes n_subject > n_components
    meaningful (and is required to exercise back-reconstruction's subject-specific path).
    The phase stays exactly phi(v), so the phase-quality mask is unaffected.
    """
    N, Vsig = Smaps.shape
    tc = rng.standard_normal((T, N))
    base = 100.0 + 10.0 * rng.random(Vsig)
    M = base[None, :] + 5.0 * (tc @ Smaps)  # positive magnitude
    if noise:
        M = M + noise * rng.standard_normal((T, Vsig))
    assert np.all(M > 0)
    return M * np.exp(1j * phi)[None, :]


def test_end_to_end_group_complex_ica():
    rng = np.random.default_rng(21)
    N, Vsig, Vnoise, T, n_sub = 3, 1000, 300, 60, 4

    Smaps = rng.standard_normal((N, Vsig)) * np.abs(rng.standard_normal((N, Vsig))) ** 1.5
    Smaps /= np.abs(Smaps).max()
    phi = (rng.random(Vsig) * 20 - 10) / 180 * np.pi  # static phase, +/-10 deg

    subjects = []
    for _ in range(n_sub):
        sig = _subject(rng, Smaps, phi, T)  # (T, Vsig)
        noise = (0.5 + rng.random((T, Vnoise))) * np.exp(1j * 2 * np.pi * rng.random((T, Vnoise)))
        subjects.append(np.concatenate([sig, noise], axis=1))  # (T, V)

    res = run_complex_ica(subjects, n_components=N, estimator='cebm', rng=np.random.default_rng(0))

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
    )  # (N, V)
    Strue = StrueFull[:, res.mask]  # (N, V_masked)
    perm, corr = match_sources(res.S_group, Strue)
    assert len(set(perm)) == N
    assert np.all(corr > 0.8)

    # phase correction left the maps concentrated on the real axis
    imag_frac = (res.S_group.imag**2).sum(axis=1) / (np.abs(res.S_group) ** 2).sum(axis=1)
    assert np.all(imag_frac < 0.15)

    # back-reconstruction produced one map set per subject, aligned to the group
    assert len(res.subjects) == n_sub
    for S_i, _A_i in res.subjects:
        assert S_i.shape == res.S_group.shape
        p_i, c_i = match_sources(S_i, res.S_group)
        assert list(p_i) == list(range(N))  # already aligned: identity permutation
        assert np.all(c_i > 0.8)


def test_back_reconstruction_is_subject_specific():
    """Guard against a silently collapsed back-reconstruction.

    GICA back-reconstruction must use each subject's ACTUAL reduced data:
    `S_i = pinv(Bi) @ Y_i`. Writing the tempting `S_i = pinv(Bi) @ (Bi @ S_group)` instead
    collapses algebraically to `S_group` for EVERY subject (since `pinv(Bi) @ Bi = I`),
    carrying zero subject-specific variance -- a silent no-op.

    The end-to-end test above CANNOT catch that: it builds every subject from the same
    maps, so `S_i == S_group` there by construction and the no-op would still pass. Here
    each subject gets its own perturbed maps, and we require the back-reconstructed maps to
    explain that subject better than the group maps do -- which is false if S_i collapses.
    """
    rng = np.random.default_rng(31)
    N, Vsig, Vnoise, T, n_sub = 3, 1000, 300, 80, 4

    Smaps = rng.standard_normal((N, Vsig)) * np.abs(rng.standard_normal((N, Vsig))) ** 1.5
    Smaps /= np.abs(Smaps).max()
    phi = (rng.random(Vsig) * 20 - 10) / 180 * np.pi

    subjects, Strue_i = [], []
    for _ in range(n_sub):
        # this subject's OWN maps: the group maps plus a real per-subject perturbation
        Si = Smaps + 0.5 * (
            rng.standard_normal((N, Vsig)) * np.abs(rng.standard_normal((N, Vsig))) ** 1.5
        )
        Si /= np.abs(Si).max()
        # subject-specific magnitude noise lifts the rank above N, so the reduction can
        # legitimately keep n_subject=6 > n_components=3 components
        sig = _subject(rng, Si, phi, T, noise=2.0)
        noise = (0.5 + rng.random((T, Vnoise))) * np.exp(1j * 2 * np.pi * rng.random((T, Vnoise)))
        subjects.append(np.concatenate([sig, noise], axis=1))
        Strue_i.append(
            np.concatenate(
                [Si * np.exp(1j * phi)[None, :], np.zeros((N, Vnoise), dtype=np.complex128)],
                axis=1,
            )
        )

    # n_subject > n_components, so each subject's reduced data retains subject-specific
    # structure for back-reconstruction to recover.
    res = run_complex_ica(
        subjects, n_components=N, n_subject=6, estimator='cebm', rng=np.random.default_rng(0)
    )

    for (S_i, _A_i), St in zip(res.subjects, Strue_i, strict=True):
        St_m = St[:, res.mask]
        _, corr_subject = match_sources(S_i, St_m)
        _, corr_group = match_sources(res.S_group, St_m)
        # the back-reconstructed maps must explain THIS subject better than the group maps
        assert corr_subject.mean() > corr_group.mean()


def test_pipeline_runs_with_pdv_mask():
    """The GIFT PDV mask is selectable end-to-end and recovers the group maps.

    PDV keys on SPATIAL phase smoothness, so signal is a large smooth-phase region and
    noise is a compact random-phase pocket -- unlike the default temporal measure, which
    keys on per-voxel phase stability over time. A small smooth block would fail here:
    its sharp boundary against the noise reads as high variance and the per-slice erosion
    consumes its interior.
    """
    rng = np.random.default_rng(11)
    N, T, n_sub = 3, 50, 4
    dims = (40, 40, 2)
    x, y, s = dims
    V = x * y * s

    noise = np.zeros((x, y, s), dtype=bool)
    noise[8:20, 8:20, :] = True  # compact random-phase pocket
    good = (~noise).reshape(V)
    noise = noise.reshape(V)
    Vsig = int(good.sum())

    Smaps = rng.standard_normal((N, Vsig)) * np.abs(rng.standard_normal((N, Vsig))) ** 1.5
    Smaps /= np.abs(Smaps).max()
    phi = 0.02 * rng.standard_normal(Vsig)  # small, smooth static phase over the region

    subjects = []
    for _ in range(n_sub):
        tc = rng.standard_normal((T, N))
        base = 100.0 + 10.0 * rng.random(Vsig)
        M = base[None, :] + 5.0 * (tc @ Smaps)
        Z = np.zeros((T, V), dtype=np.complex128)
        Z[:, good] = M * np.exp(1j * phi)[None, :]  # smooth-phase signal region
        Z[:, noise] = np.exp(1j * np.pi * (2 * rng.random((T, V - Vsig)) - 1))  # random phase
        subjects.append(Z)

    res = run_complex_ica(
        subjects,
        n_components=N,
        estimator='cebm',
        mask_method='pdv',
        dims=dims,
        rng=np.random.default_rng(0),
    )

    # the PDV mask kept the smooth region and dropped the random-phase pocket
    assert res.mask[good].mean() > 0.8
    assert res.mask[noise].mean() < 0.05

    # and the decomposition still recovered the true maps under that mask
    Strue = np.zeros((N, V), dtype=np.complex128)
    Strue[:, good] = Smaps * np.exp(1j * phi)[None, :]
    _perm, corr = match_sources(res.S_group, Strue[:, res.mask])
    assert np.all(corr > 0.8)
