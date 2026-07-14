"""The complete complex ICA pipeline, wired end to end."""

from dataclasses import dataclass

import numpy as np

from .estimators import get_estimator
from .group import back_reconstruct, two_stage_pca
from .phase_correct import align_to_reference, correct_phase
from .phase_mask import phase_quality_mask


@dataclass
class PipelineResult:
    S_group: np.ndarray             # (N, V_masked) group maps
    A_group: np.ndarray             # group mixing
    mask: np.ndarray                # (V,) boolean
    subjects: list                  # [(S_i, A_i), ...] back-reconstructed, group-aligned


def run_complex_ica(subject_data, n_components, estimator="cebm", mag_mask=None,
                    n_subject=None, rng=None, **estimator_kwargs):
    """Run group complex ICA over a list of (T_i, V) complex subject arrays."""
    if rng is None:
        rng = np.random.default_rng()
    if n_subject is None:
        n_subject = n_components

    # 1. phase-quality mask, computed on the concatenated complex data (V, T_total)
    Z = np.concatenate([np.asarray(x, dtype=np.complex128) for x in subject_data],
                       axis=0).T                                   # (V, T_total)
    mask, _, _ = phase_quality_mask(Z, mag_mask=mag_mask)

    masked = [np.asarray(x, dtype=np.complex128)[:, mask] for x in subject_data]

    # 2. two-stage complex PCA reduction
    Xg, reduced, whiteners, W_group = two_stage_pca(masked, n_subject, n_components)

    # 3. complex ICA
    est = get_estimator(estimator)
    res = est(Xg, rng=rng, **estimator_kwargs) if estimator == "cebm" \
        else est(Xg, **estimator_kwargs)

    # 4. phase-ambiguity correction on the group maps
    S_group, A_group, _ = correct_phase(res.S, res.A)

    # 5. back-reconstruct per subject, then align each to the group reference
    subjects = []
    for S_i, A_i in back_reconstruct(S_group, A_group, reduced, whiteners, W_group):
        S_i, _ = align_to_reference(S_i, S_group)
        subjects.append((S_i, A_i))

    return PipelineResult(S_group=S_group, A_group=A_group, mask=mask, subjects=subjects)
