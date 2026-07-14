"""Phase-quality mask (quality-map thresholding).

Phase is only trustworthy where SNR is high: in brain voxels the complex time
series points in a nearly consistent direction over time, while in noise voxels the
phase wanders. Temporal phase stability is therefore a proxy for SNR.

Reference: Rodriguez, Correa, Eichele, Calhoun & Adali (2011), J. Signal Process.
Syst. 65:497-508.
"""

import numpy as np
from skimage.filters import threshold_otsu


def quality_map(Z):
    """Q[v] = |sum_t Z[v, t]| / sum_t |Z[v, t]|, in [0, 1]. Z is (V, T) complex.

    Invariant to a constant per-voxel phase offset (it measures variation over time).
    """
    Z = np.asarray(Z, dtype=np.complex128)
    num = np.abs(Z.sum(axis=1))
    den = np.abs(Z).sum(axis=1) + np.finfo(np.float64).eps
    return num / den


def otsu_threshold(x):
    """Otsu threshold of a 1-D array."""
    return float(threshold_otsu(np.asarray(x, dtype=np.float64)))


def phase_quality_mask(Z, mag_mask=None):
    """(mask, Q, tau). Intersects a magnitude/brain mask with Q > Otsu(Q)."""
    Z = np.asarray(Z, dtype=np.complex128)
    V = Z.shape[0]
    if mag_mask is None:
        mag_mask = np.ones(V, dtype=bool)
    mag_mask = np.asarray(mag_mask, dtype=bool).ravel()

    Q = quality_map(Z)
    tau = otsu_threshold(Q[mag_mask])
    mask = mag_mask & (Q > tau)
    return mask, Q, tau
