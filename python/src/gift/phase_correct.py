"""Phase-ambiguity correction (the complex-domain analogue of real ICA's sign fix).

Complex ICA recovers each source only up to a complex scalar c_k * exp(i*theta_k):

    X = A @ S  <=>  X = (A @ inv(D)) @ (D @ S),   D = diag(c_k * exp(i*theta_k))

so the split of a component's energy between real and imaginary parts is arbitrary.
Convention: rotate each component so its energy is maximally concentrated in the real
part. The voxel values form an elongated cloud ~ r(v) * exp(i*alpha); squaring maps the
+/-alpha line ambiguity to the single angle 2*alpha, giving the closed form
theta = -0.5 * angle(sum(s**2)).

Reference: Rodriguez, Calhoun & Adali (2012), Pattern Recognition 45:2050-2063.
"""

import numpy as np
from scipy.stats import skew


def correct_phase(S, A, mask=None):
    """Rotate each component onto the real axis; apply the inverse to A.

    S: (N, V) complex sources. A: (M, N) complex mixing. Returns (S, A, theta).
    A @ S is preserved exactly.
    """
    S = np.array(S, dtype=np.complex128, copy=True)
    A = np.array(A, dtype=np.complex128, copy=True)
    N, V = S.shape
    if mask is None:
        mask = np.ones(V, dtype=bool)
    mask = np.asarray(mask, dtype=bool).ravel()

    theta = np.zeros(N, dtype=np.float64)
    for k in range(N):
        sm = S[k, mask]
        th = -0.5 * np.angle(np.sum(sm**2))  # orient the major axis to the real axis
        sk = S[k] * np.exp(1j * th)
        # residual pi ambiguity: fix the direction by real-part skewness
        if skew(sk[mask].real) < 0:
            th += np.pi
            sk = -sk
        S[k] = sk
        A[:, k] = A[:, k] * np.exp(-1j * th)
        theta[k] = th
    return S, A, theta


def align_to_reference(S, S_ref):
    """Align each subject component to a reference (e.g. the group/aggregate map).

    Per-subject phase corrections are independent, so without this the residual
    rotations reintroduce non-physiological variance before group statistics.
    Returns (S_aligned, theta).
    """
    S = np.array(S, dtype=np.complex128, copy=True)
    S_ref = np.asarray(S_ref, dtype=np.complex128)
    N = S.shape[0]
    theta = np.zeros(N, dtype=np.float64)
    for k in range(N):
        th = -np.angle(np.sum(np.conj(S_ref[k]) * S[k]))
        S[k] = S[k] * np.exp(1j * th)
        theta[k] = th
    return S, theta
