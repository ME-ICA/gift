"""The complete complex ICA pipeline, wired end to end."""

from dataclasses import dataclass

import numpy as np

from .estimators import get_estimator
from .group import back_reconstruct, two_stage_pca
from .phase_correct import align_to_reference, correct_phase
from .phase_mask import TEMPORAL, combine_subject_masks, phase_quality_mask


@dataclass
class PipelineResult:
    """The output of :func:`run_complex_ica`.

    Attributes
    ----------
    S_group : numpy.ndarray of complex128, shape (N, V_masked)
        The group component maps, over the masked voxels only. Use
        :func:`gift.driver.unmask` to expand them back to full volume shape.
    A_group : numpy.ndarray of complex128, shape (N, N)
        The group mixing matrix, in the *group-reduced* space. These are not subject
        time courses; see Notes.
    mask : numpy.ndarray of bool, shape (V,)
        The phase-quality mask, over all voxels. ``mask.sum() == V_masked``.
    subjects : list of (numpy.ndarray, numpy.ndarray)
        One ``(S_i, A_i)`` pair per input subject, in input order, back-reconstructed
        and phase-aligned to the group maps. ``S_i`` is ``(N, V_masked)`` subject maps,
        ``A_i`` is ``(T_i, N)`` subject time courses.

    Notes
    -----
    ``A_group`` lives in the group-reduced space and has shape ``(N, N)``, so it is not
    directly interpretable as a time course. Actual time courses, in units of the input
    timepoints, are the ``A_i`` in ``subjects``.

    Component numbering is not stable across runs or ports. ICA fixes neither the order
    nor the phase of its components, so match components by correlation, never by index.
    """

    S_group: np.ndarray
    A_group: np.ndarray
    mask: np.ndarray
    subjects: list


def run_complex_ica(
    subject_data,
    n_components,
    estimator='cebm',
    mag_mask=None,
    n_subject=None,
    rng=None,
    mask_method=TEMPORAL,
    dims=None,
    mask_agreement=0.8,
    **estimator_kwargs,
):
    """Run complex ICA over one or more complex subject arrays.

    Parameters
    ----------
    subject_data : list of array_like of complex, each of shape (T_i, V)
        One entry per subject: a complex array of ``T_i`` timepoints by ``V`` voxels.
        Subjects may differ in ``T_i`` but must agree on ``V``. For single-subject
        analysis, pass a one-element list; see Notes.
    n_components : int
        The model order ``N``: how many independent components to extract.
    estimator : {'cebm', 'nc-fastica'}, optional
        Which complex ICA estimator to use. Default is ``'cebm'`` (complex ICA by
        entropy bound minimization). Note the hyphen in ``'nc-fastica'``.
    mag_mask : array_like of bool, shape (V,), or None, optional
        A magnitude or brain mask, intersected with the phase-quality mask. Default is
        None, which considers every voxel.
    n_subject : int or None, optional
        The subject-level PCA order, i.e. how many components to retain per subject
        before concatenation. Default is None, which uses ``n_components``. Ignored in
        effect for single-subject analysis, where the two reduction stages collapse.
    rng : numpy.random.Generator or None, optional
        Random generator, used by the ``'cebm'`` estimator for its initialization.
        Default is None, which seeds a fresh generator nondeterministically. Pass an
        explicit generator for reproducible results.
    mask_method : {'temporal', 'pdv'}, optional
        Which phase-quality measure builds the mask. Default is ``'temporal'`` (per-voxel
        temporal coherence, thresholded by Otsu's method over all subjects at once).
        ``'pdv'`` is GIFT's spatial phase-derivative variance; it requires ``dims``,
        builds a mask per subject, and votes (see ``mask_agreement``). See
        :func:`gift.phase_mask.phase_quality_mask`.
    dims : tuple of int, or None, optional
        The spatial shape ``(X, Y, S)``, with ``prod(dims) == V``. Required when
        ``mask_method='pdv'``; ignored otherwise.
    mask_agreement : float, optional
        For ``mask_method='pdv'`` only: the fraction of subjects that must mark a voxel
        good for it to survive. Default is 0.8, GIFT's default.
    **estimator_kwargs
        Extra keyword arguments forwarded to the estimator.

    Returns
    -------
    PipelineResult
        The group maps, group mixing, mask, and per-subject back-reconstructions.

    Notes
    -----
    **Single-subject analysis.** Pass a one-element list::

        result = run_complex_ica([X], n_components=20, rng=np.random.default_rng(0))
        maps = result.S_group                 # (20, V_masked) component maps
        _, timecourses = result.subjects[0]   # (T, 20) component time courses

    With one subject, back-reconstruction is a no-op: ``result.subjects[0][0]`` equals
    ``result.S_group`` to within floating-point round-off, so take the maps from
    ``S_group`` and take only the time courses from ``subjects[0]``. The time courses are
    the useful half of that pair, since ``A_group`` is in the group-reduced space rather
    than in timepoints.

    The reconstruction ``A_i @ S_i`` approximates the *de-meaned, masked* input, not the
    raw input: the pipeline removes each voxel's temporal mean, and restricts to
    ``result.mask``.

    **The pipeline.** Five stages run in order:

    1. A phase-quality mask is computed and shared across subjects, dropping voxels whose
       phase is untrustworthy. The default ``'temporal'`` method keys on per-voxel phase
       stability over time (one mask from all subjects concatenated); ``'pdv'`` keys on
       GIFT's spatial phase smoothness (one mask per subject, then a vote). See
       ``mask_method``.
    2. Two-stage complex PCA reduces each subject, temporally concatenates them, then
       reduces again at the group level. For a single subject this degenerates to
       ordinary PCA.
    3. The chosen complex ICA estimator runs on the group-reduced data.
    4. The group maps are phase-corrected, resolving the complex scaling ambiguity
       inherent to complex ICA, which fixes a phase for each component.
    5. Each subject is back-reconstructed and then phase-aligned to the group maps.

    Stage 4 is needed because complex ICA determines each source only up to an arbitrary
    complex scalar. Stage 5's alignment rotates each subject's maps onto the group
    reference; the subject's mixing is counter-rotated by the inverse phase, so that the
    returned pair still satisfies :math:`X_i \\approx A_i S_i`.
    """
    if rng is None:
        rng = np.random.default_rng()
    if n_subject is None:
        n_subject = n_components

    # 1. phase-quality mask.
    if mask_method == TEMPORAL:
        # One mask from all subjects' timepoints concatenated: (V, T_total).
        Z = np.concatenate([np.asarray(x, dtype=np.complex128) for x in subject_data], axis=0).T
        mask, _, _ = phase_quality_mask(Z, mag_mask=mag_mask, method=TEMPORAL)
    else:
        # GIFT's PDV scheme: mask each subject on its own, then vote. Concatenating first
        # would instead demand a voxel be good at every timepoint of every subject, which
        # is not what the reference does.
        per_subject = []
        for x in subject_data:
            Zi = np.asarray(x, dtype=np.complex128).T  # (V, T_i)
            mi, _, _ = phase_quality_mask(Zi, mag_mask=mag_mask, method=mask_method, dims=dims)
            per_subject.append(mi)
        # Each per-subject mask already carries mag_mask, so voxels outside it cannot win
        # the vote; no further intersection is needed.
        mask = combine_subject_masks(per_subject, agreement=mask_agreement)

    masked = [np.asarray(x, dtype=np.complex128)[:, mask] for x in subject_data]

    # 2. two-stage complex PCA reduction
    Xg, reduced, whiteners, W_group = two_stage_pca(masked, n_subject, n_components)

    # 3. complex ICA
    est = get_estimator(estimator)
    if estimator == 'cebm':
        res = est(Xg, rng=rng, **estimator_kwargs)
    else:
        res = est(Xg, **estimator_kwargs)

    # 4. phase-ambiguity correction on the group maps
    S_group, A_group, _ = correct_phase(res.S, res.A)

    # 5. back-reconstruct per subject, then align each to the group reference
    subjects = []
    for S_i, A_i in back_reconstruct(S_group, A_group, reduced, whiteners, W_group):
        S_i, theta_i = align_to_reference(S_i, S_group)
        # Counter-rotate A_i so the returned pair still satisfies X_i ~= A_i @ S_i:
        # A_i is (T_i, N) with components as columns, so each column k must be rotated
        # by the inverse of the phase applied to S_i's row k.
        A_i = A_i * np.exp(-1j * theta_i)[None, :]
        subjects.append((S_i, A_i))

    return PipelineResult(S_group=S_group, A_group=A_group, mask=mask, subjects=subjects)
