"""File-level driver: complex NIfTI in, complex component maps out.

`pipeline.run_complex_ica` works on in-memory `(T, V)` arrays and returns maps over the
masked voxels only. This module is the thin layer that makes the package usable on real
data: it loads GIFT's two-file complex volumes, flattens them, runs the pipeline, expands
the maps back to full volume shape, and writes them out as complex NIfTI pairs.
"""

from pathlib import Path

import nibabel as nib
import numpy as np

from .complex_io import REAL_IMAG, read_complex, write_complex
from .pipeline import run_complex_ica


def unmask(S, mask, dims):
    """Expand masked maps back to full volume shape.

    S: (N, V_masked) complex. mask: (V,) bool. dims: spatial shape, prod(dims) == V.
    Returns (N, *dims) complex, zero outside the mask.
    """
    S = np.asarray(S, dtype=np.complex128)
    mask = np.asarray(mask, dtype=bool).ravel()
    N = S.shape[0]
    V = mask.size
    if int(np.prod(dims)) != V:
        raise ValueError(f"dims {tuple(dims)} hold {int(np.prod(dims))} voxels, mask has {V}")

    full = np.zeros((N, V), dtype=np.complex128)
    full[:, mask] = S
    return full.reshape(N, *dims)


def _load_subject(pair, complex_type):
    """Load one subject's two-file complex volume as ((T, V) array, dims, affine)."""
    first, second = pair
    z = read_complex(first, second, complex_type=complex_type)     # (*dims, T)
    if z.ndim < 2:
        raise ValueError(f"expected a 4-D complex volume, got shape {z.shape}")
    dims, T = z.shape[:-1], z.shape[-1]
    affine = nib.load(str(first)).affine
    V = int(np.prod(dims))
    return z.reshape(V, T).T, dims, affine                          # (T, V)


def run_from_files(subject_pairs, n_components, out_dir, complex_type=REAL_IMAG,
                   prefix="component", naming=("R_", "I_"), **kwargs):
    """Run group complex ICA over subjects given as two-file complex NIfTI pairs.

    subject_pairs: list of (first_path, second_path). Each pair is one subject's 4-D
        complex volume (real&imaginary, or magnitude&phase per `complex_type`).
    n_components: model order.
    out_dir: directory for the written component maps (created if absent).
    kwargs: forwarded to `run_complex_ica` (e.g. `estimator`, `rng`, `n_subject`).

    Returns (result, written) where `result` is the PipelineResult and `written` is a list
    of (first_path, second_path) — one complex NIfTI pair per component.
    """
    if not subject_pairs:
        raise ValueError("no subjects given")

    loaded = [_load_subject(p, complex_type) for p in subject_pairs]
    dims = loaded[0][1]
    for _, d, _ in loaded[1:]:
        if d != dims:
            raise ValueError(f"subjects disagree on volume shape: {dims} vs {d}")
    affine = loaded[0][2]
    subject_data = [x for x, _, _ in loaded]

    result = run_complex_ica(subject_data, n_components=n_components, **kwargs)

    maps = unmask(result.S_group, result.mask, dims)                # (N, *dims)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for k, vol in enumerate(maps, start=1):
        base = f"{prefix}_{k:03d}.nii"
        first = out_dir / (naming[0] + base)
        second = out_dir / (naming[1] + base)
        write_complex(vol, first, second, affine=affine, complex_type=complex_type)
        written.append((first, second))

    return result, written
