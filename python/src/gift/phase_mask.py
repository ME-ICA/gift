"""Phase-quality masks.

Two independent measures of phase quality are offered, selected by ``method``:

``'temporal'``
    Per-voxel coherence of the complex time series. Needs no volume geometry.
``'pdv'``
    GIFT's spatial phase-derivative variance, ported from the Complex GIFT toolbox.
    Needs the 3-D volume shape.

They measure different things and are not interchangeable; see
:func:`phase_quality_mask` for how to choose.
"""

import numpy as np
from scipy.ndimage import uniform_filter
from skimage.filters import threshold_otsu
from skimage.morphology import dilation, disk, erosion

TEMPORAL = 'temporal'
PDV = 'pdv'


def wrap_phase(x):
    """Wrap phase values into ``(-pi, pi]``.

    Parameters
    ----------
    x : array_like of float
        Phase values, in radians.

    Returns
    -------
    numpy.ndarray of float64
        The wrapped values, in ``(-pi, pi]``.

    Notes
    -----
    A port of ``icatb_wrap_func.m``. That function subtracts ``2*pi`` while the value is
    ``> pi`` and adds ``2*pi`` while it is ``<= -pi``, so the interval is half-open on the
    left: exactly ``-pi`` maps to ``+pi``, not to itself. The closed form used here,
    ``x - 2*pi*ceil((x - pi) / (2*pi))``, reproduces that convention exactly, including
    the endpoint, without the reference's iterative loop.

    This differs from :func:`numpy.angle`, which returns ``[-pi, pi]`` and would map
    ``-pi`` to ``-pi``.
    """
    x = np.asarray(x, dtype=np.float64)
    return x - 2 * np.pi * np.ceil((x - np.pi) / (2 * np.pi))


def quality_map(Z):
    """Compute the temporal phase-stability quality map.

    Parameters
    ----------
    Z : array_like of complex, shape (V, T)
        Complex time series for ``V`` voxels over ``T`` timepoints.

    Returns
    -------
    numpy.ndarray of float64, shape (V,)
        The per-voxel quality value, in ``[0, 1]``. Higher means a more stable phase.

    Notes
    -----
    The quality of voxel ``v`` is the ratio of the magnitude of the summed complex
    time series to the summed magnitudes:

    .. math:: Q_v = \\frac{\\left| \\sum_t Z_{v,t} \\right|}{\\sum_t \\left| Z_{v,t} \\right|}

    This is a normalized measure of how consistently the complex time series points in
    one direction. By the triangle inequality it lies in ``[0, 1]``: it reaches 1 when
    every timepoint shares the same phase, and falls toward 0 as phases spread out and
    the vectors cancel in the sum.

    The measure is invariant to a constant per-voxel phase offset, since a common
    rotation factors out of the numerator's magnitude and leaves the denominator
    untouched. It therefore responds to phase *variation* over time, not to whatever
    arbitrary phase a voxel happens to sit at.

    The denominator carries a machine-epsilon guard so that all-zero voxels return 0
    instead of dividing by zero.

    This measure is *not* the one described in [1]_; see :func:`pdv_quality_map` for
    that. It is a temporal measure with no counterpart in the reference toolbox.

    References
    ----------
    .. [1] Rodriguez, P. A., Correa, N. M., Eichele, T., Calhoun, V. D., & Adali, T.
           (2011). Quality map thresholding for de-noising of complex-valued fMRI data
           and its application to ICA of fMRI. Journal of Signal Processing Systems,
           65(3), 497-508.
    """
    Z = np.asarray(Z, dtype=np.complex128)
    numerator = np.abs(Z.sum(axis=1))
    denominator = np.abs(Z).sum(axis=1) + np.finfo(np.float64).eps
    return numerator / denominator


def pdv_quality_map(phase, wrap=True):
    """Compute GIFT's spatial phase-derivative variance (PDV) quality map.

    Parameters
    ----------
    phase : array_like of float, shape (X, Y, ...)
        Wrapped phase in radians. The PDV is computed independently over the leading two
        axes, which must be the in-plane axes of a slice. Trailing axes (slice index,
        timepoint, ...) are broadcast over.
    wrap : bool, optional
        Whether to wrap the phase gradients into ``(-pi, pi]`` before use. Default is
        True, matching the reference's ``wrap_flag=1``, and is correct whenever ``phase``
        is wrapped rather than unwrapped.

    Returns
    -------
    numpy.ndarray of float64, shape (X, Y, ...)
        The PDV value at each position. **Lower is better**: it is a local dispersion of
        the phase gradient, so smooth (high-SNR) phase gives values near zero.

    Notes
    -----
    A port of ``icatb_val_qual_circular.m``, which implements the phase-derivative
    variance of [1]_ (eq. 3.12 of Ghiglia & Pritt's phase-unwrapping text), with the
    circular boundary conditions the reference attributes to Sofia Chavez.

    The forward differences use circular wrap-around, so that

    .. math:: FY_{i,j} = \\varphi_{i+1,j} - \\varphi_{i,j}

    with the last row taking ``phase[0] - phase[-1]``, and likewise for ``FX`` along the
    second axis. The gradients are then wrapped, and at each position the quality is

    .. math:: Q_{i,j} = \\frac{\\sqrt{x_{sum}} + \\sqrt{y_{sum}}}{k^2}

    where each sum runs over the ``k x k`` (``k = 3``) circular neighbourhood of
    deviations from that neighbourhood's mean gradient.

    The reference computes those sums with an explicit per-pixel loop. This port uses the
    algebraic identity ``sum((u - mean)^2) = k^2 * (E[u^2] - E[u]^2)`` over the same
    neighbourhood, which turns the loop into two uniform filters and is exact rather than
    approximate. The variance is clamped at zero, since catastrophic cancellation in that
    identity can otherwise produce a small negative value whose square root is NaN.

    Because the measure is built from *differences* of neighbouring phases, a constant
    phase offset added to the whole slice cancels and leaves the map unchanged.

    References
    ----------
    .. [1] Rodriguez, P. A., Correa, N. M., Eichele, T., Calhoun, V. D., & Adali, T.
           (2011). Quality map thresholding for de-noising of complex-valued fMRI data
           and its application to ICA of fMRI. Journal of Signal Processing Systems,
           65(3), 497-508.
    """
    phase = np.asarray(phase, dtype=np.float64)
    if phase.ndim < 2:
        raise ValueError(
            f'pdv_quality_map: phase needs at least 2 (in-plane) axes, got shape {phase.shape}'
        )

    # Forward differences with circular wrap-around, matching the reference's diff() plus
    # explicit last-row/last-column periodic term.
    grad_y = np.roll(phase, -1, axis=0) - phase
    grad_x = np.roll(phase, -1, axis=1) - phase
    if wrap:
        grad_y = wrap_phase(grad_y)
        grad_x = wrap_phase(grad_x)

    # 3x3 circular mean over the in-plane axes only.
    size = [3, 3] + [1] * (phase.ndim - 2)
    k2 = 9.0

    def _neighborhood_dispersion(g):
        mean_g = uniform_filter(g, size=size, mode='wrap')
        mean_g2 = uniform_filter(g * g, size=size, mode='wrap')
        var = np.maximum(mean_g2 - mean_g * mean_g, 0.0)
        return np.sqrt(k2 * var)

    return (_neighborhood_dispersion(grad_x) + _neighborhood_dispersion(grad_y)) / k2


def pdv_mask(phase, pdv_threshold=0.2, erode_radius=2, dilate_radius=2, wrap=True):
    """Build one subject's PDV mask from its 4-D phase volume.

    Parameters
    ----------
    phase : array_like of float, shape (X, Y, S, T)
        Wrapped phase in radians: two in-plane axes, a slice axis, and a time axis.
    pdv_threshold : float, optional
        Voxels are good where the PDV is ``<= pdv_threshold``. Default is 0.2, the
        reference's default.
    erode_radius, dilate_radius : int, optional
        Radii of the disk used for the per-slice morphological opening. Defaults are 2
        and 2, the reference's defaults. Set either to 0 to skip that step.
    wrap : bool, optional
        Forwarded to :func:`pdv_quality_map`. Default is True.

    Returns
    -------
    numpy.ndarray of bool, shape (X, Y, S)
        The per-subject mask.

    Notes
    -----
    A port of ``icatb_qmpd_quality_mask_run.m``, which thresholds the 4-D PDV map,
    combines over time with a logical **AND** -- a voxel must be good at *every*
    timepoint, not merely on average -- and then opens each slice to drop isolated
    voxels that squeaked past the threshold.

    .. warning::

       The morphological opening uses :func:`skimage.morphology.disk`, which is a
       Euclidean disk. MATLAB's ``strel('disk', R)`` defaults to a periodic-line
       decomposition that yields a *different* neighbourhood for the same radius, so
       masks can differ by a voxel or two at object boundaries. Everything else in this
       function is exact; this step is the one deliberate approximation and it has not
       been checked against MATLAB.
    """
    phase = np.asarray(phase, dtype=np.float64)
    if phase.ndim != 4:
        raise ValueError(f'pdv_mask: phase must be 4-D (X, Y, S, T), got shape {phase.shape}')

    quality = pdv_quality_map(phase, wrap=wrap)  # (X, Y, S, T)
    good = quality <= pdv_threshold
    mask = good.all(axis=3)  # AND over time

    if erode_radius > 0 or dilate_radius > 0:
        opened = np.zeros_like(mask)
        for s in range(mask.shape[2]):
            plane = mask[:, :, s]
            if erode_radius > 0:
                plane = erosion(plane, disk(erode_radius))
            if dilate_radius > 0:
                plane = dilation(plane, disk(dilate_radius))
            opened[:, :, s] = plane
        mask = opened

    return mask


def combine_subject_masks(masks, agreement=0.8):
    """Combine per-subject masks by a fractional-agreement vote.

    Parameters
    ----------
    masks : sequence of array_like of bool
        One mask per subject, all the same shape.
    agreement : float, optional
        Fraction of subjects that must mark a voxel good for it to survive. Default is
        0.8, the reference's ``or_mask_thresh = 0.8*length(phase_files)``.

    Returns
    -------
    numpy.ndarray of bool
        The combined mask, of the same shape as each input.

    Notes
    -----
    A port of the mask-merging in ``icatb_preproc_complex_data.m``. The reference
    compares a subject count against ``0.8*n_subjects`` with ``>=``, so the threshold is
    a raw count and is *not* rounded; this port reproduces that comparison exactly.
    """
    masks = [np.asarray(m, dtype=bool) for m in masks]
    if not masks:
        raise ValueError('combine_subject_masks: no masks given')
    shapes = {m.shape for m in masks}
    if len(shapes) != 1:
        raise ValueError(f'combine_subject_masks: masks disagree on shape: {sorted(shapes)}')

    counts = np.sum(masks, axis=0)
    return counts >= agreement * len(masks)


def otsu_threshold(x):
    """Compute Otsu's threshold for a 1-D array.

    Parameters
    ----------
    x : array_like of float, shape (n,)
        The values to threshold.

    Returns
    -------
    float
        The threshold value.

    Notes
    -----
    Otsu's method assumes the values are drawn from two classes and picks the threshold
    that minimizes the within-class variance, equivalently maximizing the between-class
    variance. Here the two classes are high-quality (brain) and low-quality (noise)
    voxels, so no quality cutoff has to be chosen by hand.
    """
    return float(threshold_otsu(np.asarray(x, dtype=np.float64)))


def phase_quality_mask(
    Z,
    mag_mask=None,
    method=TEMPORAL,
    dims=None,
    pdv_threshold=0.2,
    erode_radius=2,
    dilate_radius=2,
):
    """Build a mask of voxels whose phase is stable enough to trust.

    Parameters
    ----------
    Z : array_like of complex, shape (V, T)
        Complex time series for ``V`` voxels over ``T`` timepoints.
    mag_mask : array_like of bool, shape (V,), or None, optional
        A magnitude or brain mask to intersect with. Default is None, which includes
        every voxel.
    method : {'temporal', 'pdv'}, optional
        Which quality measure to use. Default is ``'temporal'``. See Notes.
    dims : tuple of int, or None, optional
        The spatial shape ``(X, Y, S)``, with ``prod(dims) == V``. Required for
        ``'pdv'``, which needs slice geometry; ignored by ``'temporal'``.
    pdv_threshold : float, optional
        PDV cutoff; used only by ``'pdv'``. Default is 0.2.
    erode_radius, dilate_radius : int, optional
        Opening radii; used only by ``'pdv'``. Defaults are 2 and 2.

    Returns
    -------
    mask : numpy.ndarray of bool, shape (V,)
        ``mag_mask`` intersected with the voxels this method considers good.
    Q : numpy.ndarray of float64, shape (V,)
        The quality map over all ``V`` voxels. For ``'temporal'`` this is the per-voxel
        coherence (higher is better). For ``'pdv'`` it is the PDV **averaged over time**
        (lower is better) -- a summary for inspection only, since the mask itself is
        built from the un-averaged 4-D map.
    tau : float
        The threshold applied. Otsu's, for ``'temporal'``; ``pdv_threshold``, for
        ``'pdv'``.

    Raises
    ------
    ValueError
        If ``method`` is unrecognized, or if ``'pdv'`` is used without a ``dims`` whose
        product matches ``V``.

    Notes
    -----
    Phase is only trustworthy where SNR is high, but the two methods reach that
    conclusion from different evidence and are **not** interchangeable:

    ``'temporal'`` asks whether one voxel's phase holds still over time. It needs no
    volume geometry, works on a single timepoint-by-voxel array, and picks its threshold
    with Otsu's method, so it adapts to the data's own scale. It has no counterpart in
    the reference toolbox.

    ``'pdv'`` asks whether phase is spatially smooth *within a slice*, which is the
    quantity [1]_ actually describes and the Complex GIFT toolbox actually computes. It
    requires ``dims``, applies the reference's fixed 0.2 cutoff, demands a voxel be good
    at every timepoint, and morphologically opens each slice. Use it when parity with
    MATLAB matters.

    For ``'temporal'``, the threshold is computed on the quality values *inside*
    ``mag_mask`` rather than on the whole volume. Out-of-brain voxels would otherwise
    dominate the histogram that Otsu's method splits, dragging the threshold toward the
    noise class.

    This function masks one array. The reference combines *subjects* by masking each
    separately and keeping voxels good in at least 80% of them; see
    :func:`combine_subject_masks`.

    References
    ----------
    .. [1] Rodriguez, P. A., Correa, N. M., Eichele, T., Calhoun, V. D., & Adali, T.
           (2011). Quality map thresholding for de-noising of complex-valued fMRI data
           and its application to ICA of fMRI. Journal of Signal Processing Systems,
           65(3), 497-508.
    """
    Z = np.asarray(Z, dtype=np.complex128)
    V = Z.shape[0]
    if mag_mask is None:
        mag_mask = np.ones(V, dtype=bool)
    mag_mask = np.asarray(mag_mask, dtype=bool).ravel()

    if method == TEMPORAL:
        Q = quality_map(Z)
        tau = otsu_threshold(Q[mag_mask])
        return mag_mask & (Q > tau), Q, tau

    if method == PDV:
        if dims is None:
            raise ValueError("phase_quality_mask: method='pdv' requires dims=(X, Y, S)")
        if int(np.prod(dims)) != V:
            raise ValueError(
                f'phase_quality_mask: dims {tuple(dims)} hold {int(np.prod(dims))} voxels, '
                f'but Z has {V}'
            )
        phase = np.angle(Z).reshape(*dims, -1)  # (X, Y, S, T)
        mask = pdv_mask(
            phase,
            pdv_threshold=pdv_threshold,
            erode_radius=erode_radius,
            dilate_radius=dilate_radius,
        )
        Q = pdv_quality_map(phase).mean(axis=3).reshape(V)
        return mag_mask & mask.reshape(V), Q, float(pdv_threshold)

    raise ValueError(f'phase_quality_mask: unknown method {method!r}; use {TEMPORAL!r} or {PDV!r}')
