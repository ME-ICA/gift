"""Group complex ICA: two-stage PCA reduction and GICA back-reconstruction.

Follows GIFT's standard scheme: reduce each subject, temporally concatenate, reduce
again at the group level, run ICA on the group-reduced data, then back-reconstruct
per-subject maps and time courses.
"""

import numpy as np

from .whiten import whiten_hermitian


def two_stage_pca(subject_data, n_subject, n_group):
    """Subject-level then group-level complex PCA (temporal concatenation).

    subject_data: list of (T_i, V) complex arrays (already masked).
    Returns (Xg (n_group, V), reduced [list of (n_subject, V)],
             whiteners [list of (n_subject, T_i)], W_group (n_group, n_sub*n_subject)).
    """
    reduced, whiteners = [], []
    for Xi in subject_data:
        Xi = np.asarray(Xi, dtype=np.complex128)  # (T_i, V)
        # Remove the per-VOXEL TEMPORAL mean: this strips the static complex baseline
        # image (the anatomy / B0 background), which is constant over time and would
        # otherwise appear as a huge rank-1 nuisance direction and consume one of the
        # n_subject PCA slots, silently discarding a real source.
        #
        # Note `whiten_hermitian` internally removes the mean along its own axis=1,
        # which here is the VOXEL axis (a per-timepoint spatial de-mean) - a different
        # thing entirely. It does NOT remove the static baseline. Both are needed.
        #
        # This must happen AFTER the phase-quality mask, never before: the mask keys on
        # each voxel's phase being stable over time, and the dominant static baseline is
        # exactly what makes it stable.
        Xi = Xi - Xi.mean(axis=0, keepdims=True)  # per-voxel temporal mean
        Yi, W_i, _ = whiten_hermitian(Xi, n_components=n_subject)  # (n_subject, V)
        reduced.append(Yi)
        whiteners.append(W_i)

    stacked = np.concatenate(reduced, axis=0)  # (n_sub*n_subject, V)
    Xg, W_group, _ = whiten_hermitian(stacked, n_components=n_group)
    return Xg, reduced, whiteners, W_group


def back_reconstruct(S_group, A_group, reduced, whiteners, W_group):
    """GICA back-reconstruction to subject-specific maps and time courses.

    The group model is Xg = A_group @ S_group in the group-reduced space. Undo the group
    whitener to express the group mixing in the STACKED subject-reduced space:

        B = pinv(W_group) @ A_group          # (n_sub*n_subject, N)

    and partition B by subject into Bi (n_subject, N). Bi maps the sources into subject
    i's reduced space, so subject i's own maps come from projecting that subject's ACTUAL
    reduced data through it:

        S_i = pinv(Bi) @ Y_i                 # (N, V)

    Using Y_i (the real data) rather than Bi @ S_group is the whole point - the latter
    would collapse to S_group for every subject and carry no subject-specific variance.

    Returns a list of (S_i (N, V), A_i (T_i, N)).
    """
    n_sub = len(reduced)
    B = np.linalg.pinv(W_group) @ A_group  # (n_sub*n_subject, N)
    blocks = np.split(B, n_sub, axis=0)  # each (n_subject, N)

    out = []
    for Y_i, W_i, Bi in zip(reduced, whiteners, blocks, strict=True):
        S_i = np.linalg.pinv(Bi) @ Y_i  # subject-specific maps (N, V)
        A_i = np.linalg.pinv(W_i) @ Bi  # subject time courses (T_i, N)
        out.append((S_i, A_i))
    return out
