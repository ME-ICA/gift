"""Complex ICA by entropy bound minimization (complex ICA-EBM).

Ported from GroupICAT/icatb/icatb_analysis_functions/icatb_algorithms/complex_ica/
complex_ICA_EBM.m (functions ``CEBM``, ``complex_ICA_EBM_north``, ``pre_processing``
and ``inv_sqrtmH``).

Reference:
    Xi-Lin Li and Tulay Adali, "Complex independent component analysis by entropy
    bound minimization," IEEE Trans. Circuits and Systems I, 57(7):1417-1430,
    July 2010.

Copyright (C) 2023 MLSP Lab (original MATLAB); this Python translation.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
details: <https://www.gnu.org/licenses/>.

Deliberate deviations from the MATLAB reference
-----------------------------------------------
* **Injected RNG.** The reference draws from MATLAB's *global* RNG state
  (``randn`` for the initial ``W``, and ``randn``/``rand`` inside the optimizer),
  so its runs are not reproducible without touching global state. Here every draw
  comes from an injected ``numpy.random.Generator``, so a run is reproducible given
  a seeded ``rng``. Consequently this port cannot - and is not meant to - reproduce
  MATLAB's ``W`` elementwise; it is validated on separation quality.
* **type2 (the second kind of entropy bound) is not ported.** ``CEBM`` calls the
  optimizer with ``type2 = 0``, so the whole ``if type2`` branch (and the
  ``complex_nf_table``) is unreachable from the only entry point. The only caller
  that would set ``type2 = 1`` is the "refinement" pass, which is commented out in
  the reference (complex_ICA_EBM.m:241).
"""

from pathlib import Path

import numpy as np

from ..nf_table import load_nf_table, simplified_ppval
from .base import EstimatorResult

# parents[4] walks src/gift/estimators/ -> src/ -> python/ -> the repo root.
_DEFAULT_NF_TABLE = Path(__file__).resolve().parents[4] / 'complex_ica_fixtures' / 'nf_table.mat'

# The reference allocates 8 nonlinearities but only ever fills/uses 1, 3, 5, 7
# (1-based). Entries 2, 4, 6, 8 stay at zero and still take part in the max(), so a
# zero bound (the Gaussian case) can win. Kept faithfully.
_K_REAL = 8
_USED = (1, 3, 5, 7)  # 1-based nonlinearity indices, as in the MATLAB


def _pseudo_cov(X):
    """Pseudo-covariance E[x x^T] -- PLAIN transpose, NOT the conjugate transpose.

    MATLAB: C = Xc*Xc.'/T  (`.'` is the plain transpose). This is the quantity that
    carries the noncircularity CEBM exploits; using `.conj().T` here silently
    destroys it while still appearing to work.
    """
    return X @ X.T / X.shape[1]


def _inv_sqrtm_h(B):
    """Inverse matrix square root of a Hermitian matrix (MATLAB ``inv_sqrtmH``)."""
    d, V = np.linalg.eigh(B)
    return (V * (1.0 / np.sqrt(d))[None, :]) @ V.conj().T


def _pre_processing(X):
    """MATLAB ``pre_processing``: remove DC, then spatial pre-whitening.

    Returns ``(Xc, P)`` with ``Xc = P @ (X - mean)``.
    """
    X = np.asarray(X, dtype=np.complex128)
    T = X.shape[1]
    X = X - X.mean(axis=1, keepdims=True)
    R = X @ X.conj().T / T  # CONJUGATE transpose: the covariance
    P = _inv_sqrtm_h(R)
    return P @ X, P


def _sym_decorrelate(W):
    """MATLAB ``inv(sqrtm(W*W'))*W``. ``W @ W'`` is Hermitian, so eigh is exact."""
    return _inv_sqrtm_h(W @ W.conj().T) @ W


def _ne_bound_g1(nf, EG):
    """Negentropy bound for G1 = x^4: clamp the argument, no slope extrapolation."""
    if EG < nf.min_EGx:
        return simplified_ppval(nf.pp, nf.min_EGx)
    if EG > nf.max_EGx:
        return simplified_ppval(nf.pp, nf.max_EGx)
    return simplified_ppval(nf.pp, EG)


def _ne_bound_slope(nf, EG):
    """Negentropy bound for G3/G5/G7: linear (abs-valued) extrapolation outside the
    table, exactly as in the reference."""
    if EG < nf.min_EGx:
        d = simplified_ppval(nf.pp_slope, nf.min_EGx) * (EG - nf.min_EGx)
        return simplified_ppval(nf.pp, nf.min_EGx) + abs(d)
    if EG > nf.max_EGx:
        d = simplified_ppval(nf.pp_slope, nf.max_EGx) * (EG - nf.max_EGx)
        return simplified_ppval(nf.pp, nf.max_EGx) + abs(d)
    return simplified_ppval(nf.pp, EG)


def _standardize(z, T):
    """The reference's per-component whitening of the real/imaginary parts.

    Returns the bundle of statistics that both the SEA loop and the optimizer need.
    """
    z_real = z.real
    z_imag = z.imag
    sigma_R2 = np.sum(z_real**2) / T
    sigma_I2 = np.sum(z_imag**2) / T
    sigma_R = np.sqrt(sigma_R2)
    rho = np.sum(z_real * z_imag) / T
    Delta1 = sigma_R2 * sigma_I2 - rho**2
    sqrt_D1 = np.sqrt(Delta1)
    u = z_real / sigma_R
    v = sigma_R * z_imag / sqrt_D1 - rho * z_real / sigma_R / sqrt_D1
    return {
        'z_real': z_real,
        'z_imag': z_imag,
        'sigma_R2': sigma_R2,
        'sigma_I2': sigma_I2,
        'sigma_R': sigma_R,
        'rho': rho,
        'Delta1': Delta1,
        'u': u,
        'v': v,
    }


def _bounds(nf, x, T):
    """NE_Bound / EG vectors (length 8, 1-based semantics) for one real signal."""
    xx = x * x
    sign_x = np.sign(x)
    abs_x = sign_x * x

    NE = np.zeros(_K_REAL)
    EG = np.zeros(_K_REAL)

    # G1 = x^4
    EG[0] = np.sum(xx * xx) / T
    NE[0] = _ne_bound_g1(nf['nf1'], EG[0])
    # G3 = |x|/(1+|x|)
    EG[2] = np.sum(abs_x / (1 + abs_x)) / T
    NE[2] = _ne_bound_slope(nf['nf3'], EG[2])
    # G5 = x*|x|/(10+|x|)
    EG[4] = np.sum(x * abs_x / (10 + abs_x)) / T
    NE[4] = _ne_bound_slope(nf['nf5'], EG[4])
    # G7 = x/(1+x^2)
    EG[6] = np.sum(x / (1 + xx)) / T
    NE[6] = _ne_bound_slope(nf['nf7'], EG[6])

    return NE, EG, xx, sign_x, abs_x


def _clip(nf, value):
    """MATLAB ``min(max(EG, min_EGx), max_EGx)``."""
    return min(max(value, nf.min_EGx), nf.max_EGx)


def _sea(Xc, nf, rng, tolerance, maxiter_sea=100, max_cost_increase_number=10):
    """The "sea" fixed-point algorithm that provides the initial guess (CEBM:55-221)."""
    N, T = Xc.shape

    W = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    W = _sym_decorrelate(W)
    # NOTE: MATLAB has value semantics; every `last_W = W` / `best_W = ...` is a copy.
    # W is mutated row-by-row in place below, so these MUST be real copies in NumPy.
    last_W = W.copy()
    best_W = W.copy()

    # PLAIN transpose: C is the PSEUDO-covariance (CEBM:59). Not the covariance.
    C = _pseudo_cov(Xc)

    min_cost = np.inf
    cost_increase_counter = 0

    for _ in range(maxiter_sea):
        cost = 0.0

        for n in range(N):
            vec = W[n]  # MATLAB v = W(n,:).'  (plain transpose)
            y = vec @ Xc  # MATLAB y = v.'*Xc
            Cv = C @ vec
            vec = Xc.conj() @ (y * y * np.conj(y)) / T - 2 * vec - (vec @ Cv) * np.conj(Cv)
            W[n] = vec  # MATLAB W(n,:) = v.'

            # --- evaluate the cost with the PRE-update row (z = y) ---
            st = _standardize(y, T)
            cost += 0.5 * np.log(st['Delta1']) + np.log(2 * np.pi) + 1

            NE_u, _, _, _, _ = _bounds(nf, st['u'], T)
            NE_v, _, _, _, _ = _bounds(nf, st['v'], T)
            cost -= NE_u.max() + NE_v.max()

        if cost < min_cost:
            min_cost = cost
            cost_increase_counter = 0
            best_W = last_W  # the W this cost was computed from
        else:
            cost_increase_counter += 1

        W = _sym_decorrelate(W)  # returns a fresh array
        if cost_increase_counter > max_cost_increase_number:
            break
        if 1 - np.min(np.abs(np.diag(W @ last_W.conj().T))) < tolerance:
            break
        last_W = W.copy()

    return best_W


def _north(X, W0, max_iter_north, mu0_north, max_cost_increase_number, stochastic_search, nf, rng):
    """``complex_ICA_EBM_north``: the nonorthogonal ICA optimizer (CEBM:269-711).

    ``type2`` is fixed at 0 (see the module docstring); the second-kind entropy bound
    is unreachable from ``CEBM`` and is not ported.
    """
    N, T = X.shape
    R_xxt = _pseudo_cov(X)  # PLAIN transpose: pseudo-covariance of the whitened data

    cost_increase_counter = 0
    mu = mu0_north
    W = W0.copy()
    # W is mutated row-by-row in place; MATLAB's value semantics require real copies.
    best_W = W.copy()
    last_W = W.copy()
    min_cost = np.inf
    max_negentropy = np.zeros(N)
    negentropy_array = np.zeros(N)

    # MATLAB's `grad` persists across loop iterations. If `switch p0` matches no case
    # (possible when every NE_Bound is negative, so the untouched zero at an even index
    # wins the max), MATLAB silently reuses the previous row's `grad`. Reproduced here.
    grad = None
    inv_Q = None

    iter_idx = 0
    while iter_idx < max_iter_north:
        cost = -np.log(np.abs(np.linalg.det(W.conj().T @ W)))

        for n in range(N):
            if N > 7:
                # Incremental inverse of the Gram matrix of W-without-row-n. Two
                # Sherman-Morrison updates implement the rank-2 row swap; exactly
                # equivalent to the explicit inverse in the else-branch below.
                if n == 0:
                    Wn = W[1:]
                    inv_Q = np.linalg.inv(Wn @ Wn.conj().T)
                else:
                    n_last = n - 1  # 0-based; MATLAB's n_last is (n-1) 1-based
                    Wn_last = np.vstack([W[:n_last], W[n_last + 1 :]])
                    w_current = W[n].conj()
                    w_last = W[n_last].conj()
                    c = Wn_last @ (w_last - w_current)
                    c[n_last] = 0.5 * (w_last.conj() @ w_last - w_current.conj() @ w_current)

                    temp1 = inv_Q @ c
                    temp2 = inv_Q[:, n_last]
                    inv_Q_plus = inv_Q - np.outer(temp1, temp2.conj()) / (1 + temp1[n_last])

                    temp1 = inv_Q_plus.conj().T @ c
                    temp2 = inv_Q_plus[:, n_last]
                    inv_Q = inv_Q_plus - np.outer(temp2, temp1.conj()) / (1 + c.conj() @ temp2)
                    inv_Q = (inv_Q + inv_Q.conj().T) / 2  # inv_Q is Hermitian

                temp1 = rng.standard_normal(N).astype(np.complex128)
                W_n = np.vstack([W[:n], W[n + 1 :]])
                h = temp1 - W_n.conj().T @ inv_Q @ W_n @ temp1
            else:
                temp1 = rng.standard_normal(N).astype(np.complex128)
                temp2 = np.vstack([W[:n], W[n + 1 :]])
                h = temp1 - temp2.conj().T @ np.linalg.solve(temp2 @ temp2.conj().T, temp2 @ temp1)

            w = W[n].conj()  # MATLAB w = W(n,:)'
            z = w.conj() @ X  # MATLAB z = w'*X  == W(n,:) @ X

            st = _standardize(z, T)
            z_real, z_imag = st['z_real'], st['z_imag']
            sigma_R2, sigma_I2 = st['sigma_R2'], st['sigma_I2']
            sigma_R, rho, Delta1 = st['sigma_R'], st['rho'], st['Delta1']
            u, v = st['u'], st['v']
            sqrt_D1 = np.sqrt(Delta1)

            cost += 0.5 * np.log(Delta1)
            negentropy_array[n] = -0.5 * np.log(Delta1)

            NE_u, EGu, uu, sign_u, abs_u = _bounds(nf, u, T)
            NE_v, EGv, vv, sign_v, abs_v = _bounds(nf, v, T)

            # argmax over the separable sum; MATLAB scans p (outer) then q (inner)
            # with a strict `>`, i.e. first-max wins. argmax matches that tie-break.
            p0 = int(np.argmax(NE_u)) + 1  # back to 1-based, as in the switch
            q0 = int(np.argmax(NE_v)) + 1
            bound_real = NE_u[p0 - 1] + NE_v[q0 - 1]

            cost -= bound_real
            negentropy_array[n] += bound_real

            grad_delta1 = (0.5 * (sigma_I2 - sigma_R2) + 1j * rho) * (R_xxt @ w.conj())

            weight = rng.random(T) if stochastic_search else np.ones(T)
            sw = weight.sum()

            # --- gradient contribution from the real part (u) ---
            if p0 in _USED:
                if p0 == 1:  # G = x^4, g = 4x^3   (note: the reference does NOT clip here)
                    vEGu = simplified_ppval(nf['nf1'].pp_slope, EGu[0])
                    gu = 4 * uu * u
                elif p0 == 3:  # G = |x|/(1+|x|), g = sign(x)/(1+|x|)^2
                    vEGu = simplified_ppval(nf['nf3'].pp_slope, _clip(nf['nf3'], EGu[2]))
                    gu = sign_u / (1 + abs_u) ** 2
                elif p0 == 5:  # G = x|x|/(10+|x|), g = |x|(20+|x|)/(10+|x|)^2
                    vEGu = simplified_ppval(nf['nf5'].pp_slope, _clip(nf['nf5'], EGu[4]))
                    gu = abs_u * (20 + abs_u) / (10 + abs_u) ** 2
                else:  # p0 == 7; G = x/(1+x^2), g = (1-x^2)/(1+x^2)^2
                    vEGu = simplified_ppval(nf['nf7'].pp_slope, _clip(nf['nf7'], EGu[6]))
                    gu = (1 - uu) / (1 + uu) ** 2

                grad = 0.5 * grad_delta1 / Delta1
                grad = grad - vEGu * (X @ (weight * gu)) / sw / 2 / sigma_R
                grad = grad - vEGu * np.sum(-weight * gu * z_real) / sw / 4 / (sigma_R**3) * (
                    R_xxt @ w.conj()
                )
            elif grad is None:
                raise RuntimeError(
                    'complex ICA-EBM: every real-part negentropy bound was negative on '
                    'the first component, so no gradient is defined (MATLAB errors here '
                    'too). The data may be degenerate.'
                )

            weight = rng.random(T) if stochastic_search else np.ones(T)
            sw = weight.sum()

            # --- gradient contribution from the imaginary part (v) ---
            if q0 in _USED:
                if q0 == 1:  # G = x^4
                    vEGv = simplified_ppval(nf['nf1'].pp_slope, EGv[0])
                    gv = 4 * vv * v
                    gv_v = 4 * vv * vv  # g(v)*v
                elif q0 == 3:  # G = |x|/(1+|x|)
                    vEGv = simplified_ppval(nf['nf3'].pp_slope, _clip(nf['nf3'], EGv[2]))
                    gv = sign_v / (1 + abs_v) ** 2
                    gv_v = abs_v / (1 + abs_v) ** 2
                elif q0 == 5:  # G = x|x|/(10+|x|)
                    vEGv = simplified_ppval(nf['nf5'].pp_slope, _clip(nf['nf5'], EGv[4]))
                    gv = abs_v * (20 + abs_v) / (10 + abs_v) ** 2
                    gv_v = gv * v
                else:  # q0 == 7; G = x/(1+x^2)
                    vEGv = simplified_ppval(nf['nf7'].pp_slope, _clip(nf['nf7'], EGv[6]))
                    gv = (1 - vv) / (1 + vv) ** 2
                    gv_v = gv * v

                grad = grad + vEGv * (X @ (weight * gv)) / sw / 2 / sigma_R / sqrt_D1 * (
                    rho + 1j * sigma_R2
                )
                grad = grad + vEGv * np.sum(weight * gv_v) / sw / 2 / Delta1 * grad_delta1
                grad = grad - vEGv * np.sum(
                    weight * gv * (z_imag + (rho / sigma_R2 + 2j) * z_real)
                ) / sw / 4 / sigma_R / sqrt_D1 * (R_xxt @ w.conj())

            grad = grad - h / (w.conj() @ h)
            grad = grad - np.real(w.conj() @ grad) * w
            grad = grad / np.linalg.norm(grad)

            w1 = w - mu * grad
            w1 = w1 / np.linalg.norm(w1)
            W[n] = w1.conj()  # MATLAB W(n,:) = w1'

        if cost < min_cost:
            min_cost = cost
            best_W = last_W  # the W this cost was computed from
            max_negentropy = negentropy_array.copy()
            cost_increase_counter = 0
        else:
            cost_increase_counter += 1

        iter_idx += 1

        if cost_increase_counter > max_cost_increase_number:
            mu_floor = 1 / 20 if stochastic_search else 1 / 200
            if mu > mu_floor:
                mu = mu / 2
                cost_increase_counter = 0
                W = best_W.copy()
                last_W = W.copy()
                continue
            break

        last_W = W.copy()

    W = best_W

    # sort the components by negentropy, descending
    index_sort = np.argsort(-max_negentropy, kind='stable')
    return W[index_sort, :]


def cebm(X, nf_table_path=None, rng=None, tol=1e-4, max_iter=None):
    """Complex ICA by entropy bound minimization (MATLAB ``CEBM``).

    Parameters
    ----------
    X : (N, T) complex array
        Mixtures.
    nf_table_path : path-like, optional
        The nonlinearity table. Defaults to the repo's
        ``complex_ica_fixtures/nf_table.mat``.
    rng : numpy.random.Generator, optional
        Source of every random draw (initial ``W``, and the stochastic search's
        ``randn``/``rand``). Defaults to ``np.random.default_rng()``. Seed it to make
        a run reproducible - the MATLAB reference cannot do this without touching
        global state.
    tol : float
        Stopping tolerance of the SEA initialization (MATLAB ``tolerance``, 1e-4).
    max_iter : int, optional
        Maximum iterations of the nonorthogonal optimizer. Defaults to the
        reference's 1000.

    Returns
    -------
    EstimatorResult
        ``W`` demixes the ORIGINAL ``X`` (the pre-whitener ``P`` is folded in, as in
        ``CEBM``: ``W = W*P``), ``A = pinv(W)``, ``S = W @ X``.
    """
    X = np.asarray(X, dtype=np.complex128)
    if X.ndim != 2:
        raise ValueError(f'X must be 2-D (N, T); got shape {X.shape}')

    rng = np.random.default_rng() if rng is None else rng
    nf = load_nf_table(_DEFAULT_NF_TABLE if nf_table_path is None else nf_table_path)
    max_iter_north = 1000 if max_iter is None else int(max_iter)

    Xc, P = _pre_processing(X)

    # initial guess
    W = _sea(Xc, nf, rng, tolerance=tol)

    # stochastic gradient search. The reference's "refinement" pass
    # (complex_ICA_EBM.m:241) is commented out there, so it is not run here either.
    W = _north(
        Xc,
        W,
        max_iter_north=max_iter_north,
        mu0_north=1 / 5,
        max_cost_increase_number=5,
        stochastic_search=True,
        nf=nf,
        rng=rng,
    )

    W = W @ P  # fold the pre-whitener back in: W now demixes the original X

    return EstimatorResult(W=W, A=np.linalg.pinv(W), S=W @ X)
