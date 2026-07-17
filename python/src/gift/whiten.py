"""Complex whitening."""

import numpy as np


def whiten_hermitian(X, n_components=None):
    """PCA-whiten complex data using the Hermitian covariance.

    Parameters
    ----------
    X : array_like of complex, shape (P, T)
        Input data, with ``P`` variables observed over ``T`` samples.
    n_components : int or None, optional
        Model order ``N``: how many principal components to retain. Default is None,
        which retains all ``P`` of them.

    Returns
    -------
    Xw : numpy.ndarray of complex128, shape (N, T)
        The whitened data, with identity Hermitian covariance.
    W_whiten : numpy.ndarray of complex128, shape (N, P)
        The whitening matrix, such that ``Xw == W_whiten @ Xc`` for mean-centered
        ``Xc``.
    W_dewhiten : numpy.ndarray of complex128, shape (P, N)
        The dewhitening matrix, which maps whitened data back to the input space.

    Raises
    ------
    ValueError
        If ``n_components`` exceeds ``P``, or if the data is rank-deficient at the
        requested model order.

    Notes
    -----
    This mirrors GIFT's ``icatb_pca_whitening.m``. It uses only the covariance
    :math:`E[xx^H]`, which is the right thing for circular (proper) sources. For
    noncircular sources, the pseudo-covariance :math:`E[xx^T]` also carries signal;
    see :func:`strong_uncorrelating_transform`.

    The covariance is eigendecomposed as :math:`R = U \\Lambda U^H`. Because ``R`` is
    Hermitian its eigenvalues are real and non-negative, so whitening is the rescaling
    :math:`W = \\Lambda^{-1/2} U^H`, which gives :math:`E[(Wx)(Wx)^H] = I`.

    Rank deficiency is rejected rather than tolerated: a retained eigenvalue at or
    below ``1e-12`` times the largest one would make :math:`\\Lambda^{-1/2}` blow up
    and quietly contaminate the result with NaNs or garbage.
    """
    X = np.asarray(X, dtype=np.complex128)
    P, T = X.shape
    N = P if n_components is None else int(n_components)

    if N > P:
        raise ValueError(
            f'whiten_hermitian: requested n_components={N} exceeds the number of '
            f'input rows P={P}; cannot whiten to more components than input dimensions.'
        )

    Xc = X - X.mean(axis=1, keepdims=True)
    R = Xc @ Xc.conj().T / T  # (P, P) Hermitian; note conjugate transpose
    d, U = np.linalg.eigh(R)  # ascending, real eigenvalues
    order = np.argsort(d)[::-1][:N]  # descending, keep top N
    d = d[order].real
    U = U[:, order]

    d_max = d.max() if d.size else 0.0
    if np.any(d <= 1e-12 * d_max):
        raise ValueError(
            f'whiten_hermitian: data is rank-deficient at the requested model order '
            f'n_components={N} (smallest retained eigenvalue {d.min():.3g} is not '
            f'comfortably positive relative to the largest {d_max:.3g}); refusing to '
            'return a silently NaN-contaminated result.'
        )

    s = np.sqrt(d)
    W_whiten = (U.conj().T) / s[:, None]  # (N, P)
    W_dewhiten = U * s[None, :]  # (P, N)
    Xw = W_whiten @ Xc
    return Xw, W_whiten, W_dewhiten


def strong_uncorrelating_transform(X):
    """Simultaneously whiten the covariance and diagonalize the pseudo-covariance.

    Parameters
    ----------
    X : array_like of complex, shape (N, T)
        Input data, with ``N`` variables observed over ``T`` samples.

    Returns
    -------
    Xs : numpy.ndarray of complex128, shape (N, T)
        The transformed data: identity Hermitian covariance and diagonal, real,
        non-negative pseudo-covariance.
    W_sut : numpy.ndarray of complex128, shape (N, N)
        The transform, such that ``Xs == W_sut @ Xc`` for mean-centered ``Xc``.

    Raises
    ------
    RuntimeError
        If the computed Takagi factor fails its unitarity check, which would mean the
        returned transform is wrong.
    ValueError
        Propagated from :func:`whiten_hermitian` if the data is rank-deficient.

    Notes
    -----
    Ordinary whitening standardizes the covariance :math:`E[xx^H]` and stops there.
    This transform additionally diagonalizes the pseudo-covariance :math:`E[xx^T]`,
    which is identically zero for circular sources but carries signal when sources are
    noncircular. Noncircular estimators such as nc-FastICA require it.

    The transform proceeds in two stages. First the data is Hermitian-whitened. The
    whitened pseudo-covariance :math:`P_c = X_w X_w^T / T` is then complex *symmetric*
    rather than Hermitian, so it admits a Takagi factorization
    :math:`P_c = V \\Lambda V^T` with ``V`` unitary and :math:`\\Lambda` real and
    non-negative. Rotating by :math:`Q = V^H` diagonalizes :math:`P_c` while leaving
    the whitened covariance at identity, since any unitary preserves it.

    Deriving the Takagi basis from an ordinary SVD :math:`P_c = U S V_{svd}^H`: because
    :math:`P_c` is symmetric, :math:`P_c = P_c^T` forces
    :math:`Z := U^H \\overline{V_{svd}}` to commute with :math:`\\mathrm{diag}(S)`, and
    ``Z`` is unitary as a product of unitaries. When the singular values are all
    distinct, that commutation forces ``Z`` to be diagonal with unit-modulus entries
    :math:`p_i`, and the Takagi vectors are :math:`V = U \\, \\mathrm{diag}(\\sqrt{p_i})`.

    That shortcut breaks down under degeneracy. Whenever ``S`` has repeated (or several
    exactly zero) singular values, ``Z`` is only forced to be *block*-diagonal within
    each degenerate group, so taking its diagonal picks up non-unit-modulus entries and
    yields a measurably non-unitary ``V``. This is not hypothetical: it occurs for
    realistic fMRI data with multiple perfectly circular components.

    The degeneracy-safe fix used here is the unitary *square root* of ``Z`` rather than
    its diagonal. For unitary ``Z``, the eigendecomposition :math:`Z = Q W Q^{-1}` has
    :math:`|w_i| = 1`, and :math:`Z^{1/2} = Q W^{1/2} Q^{-1}` is well defined at any
    eigenvalue multiplicity, because the projector onto a degenerate eigenspace does not
    depend on which eigenbasis is returned within it. Then :math:`V = U Z^{1/2}` is
    unitary and satisfies :math:`P_c = V S V^T` exactly, in both the generic and the
    degenerate case. See ``test_whiten.py::test_sut_handles_degenerate_singular_values``.

    References
    ----------
    .. [1] Eriksson, J., & Koivunen, V. (2006). Complex random vectors and ICA models:
           identifiability, uniqueness, and separability. IEEE Transactions on
           Information Theory, 52(3), 1017-1029.
    """
    X = np.asarray(X, dtype=np.complex128)
    N, T = X.shape
    Xc = X - X.mean(axis=1, keepdims=True)

    # 1. standard Hermitian whitening
    Xw, W_wh, _ = whiten_hermitian(Xc, n_components=N)

    # 2. Takagi-factorize the whitened pseudo-covariance via SVD, using the unitary
    #    square root of Z so that degenerate singular values stay safe (see Notes).
    Pc = Xw @ Xw.T / T
    U_, _s, Vh_ = np.linalg.svd(Pc)
    V_svd = Vh_.conj().T
    Z = U_.conj().T @ V_svd.conj()  # unitary; diagonal only if S has no repeats
    w, Qz = np.linalg.eig(Z)
    w = w / np.abs(w)  # guard against roundoff drift off |w| == 1
    sqrtZ = Qz @ np.diag(np.sqrt(w)) @ np.linalg.inv(Qz)
    V = U_ @ sqrtZ  # Takagi basis: Pc = V @ diag(_s) @ V.T

    N_ = V.shape[0]
    unitarity_err = np.abs(V.conj().T @ V - np.eye(N_)).max()
    if not np.isfinite(unitarity_err) or unitarity_err > 1e-6:
        raise RuntimeError(
            'strong_uncorrelating_transform: Takagi factor V failed the '
            f'unitarity check (max |V^H V - I| = {unitarity_err:.3g} > 1e-6); '
            'refusing to return a silently wrong transform.'
        )

    W_sut = V.conj().T @ W_wh
    Xs = W_sut @ Xc
    return Xs, W_sut
