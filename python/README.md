# gift

Complex-valued fMRI ICA — a Python port of GIFT's complex ICA pipeline.

Ordinary fMRI analysis throws away the phase of the reconstructed complex image and keeps
only the magnitude. This package works on the complex data directly: it keeps the voxels
whose phase is stable over time, decomposes them with a complex-valued ICA estimator, and
resolves the phase ambiguity that complex ICA leaves behind.

## Installation

Requires Python >= 3.11. From this `python/` directory:

    pip install -e .

Dependencies (numpy, scipy, nibabel, scikit-image) install automatically. For the test
suite and linter, add the extras:

    pip install -e '.[tests,dev]'

## Quickstart: single-subject complex ICA

The core entry point is `run_complex_ica`. It takes a **list** of subjects, each a complex
array of shape `(T, V)` — timepoints by voxels. For a single subject, pass a one-element
list:

```python
import numpy as np
from gift.pipeline import run_complex_ica

# X is your complex data: (T timepoints, V voxels), dtype complex
result = run_complex_ica([X], n_components=20, rng=np.random.default_rng(0))

maps = result.S_group                # (20, V_masked) component maps
_, timecourses = result.subjects[0]  # (T, 20) component time courses
mask = result.mask                   # (V,) bool: which voxels were kept
```

Two things to know about the single-subject case:

**Take the maps from `S_group` and the time courses from `subjects[0]`.** With one
subject, back-reconstruction is a no-op, so `result.subjects[0][0]` equals
`result.S_group` to within floating-point round-off. The time courses are the useful half
of that pair.

**`A_group` is not a time course.** It has shape `(N, N)` and lives in the group-reduced
space. If you want something in units of your input timepoints, it is the `A_i` from
`result.subjects`.

The reconstruction `A_i @ S_i` approximates the **de-meaned, masked** input — not your raw
input. The pipeline removes each voxel's temporal mean and restricts to `result.mask`:

```python
Xm = X[:, result.mask]
Xc = Xm - Xm.mean(axis=0, keepdims=True)
rel = np.linalg.norm(Xc - timecourses @ maps) / np.linalg.norm(Xc)   # small
```

## Group complex ICA

Pass more than one subject. Subjects may differ in their number of timepoints `T_i`, but
must agree on `V`:

```python
result = run_complex_ica(
    [X1, X2, X3, X4],       # each (T_i, V) complex
    n_components=20,        # group model order N
    n_subject=30,           # per-subject PCA order (defaults to n_components)
    estimator='cebm',
    rng=np.random.default_rng(0),
)

result.S_group     # (20, V_masked) group maps
result.mask        # (V,) bool, shared across all subjects
for S_i, A_i in result.subjects:
    ...            # S_i: (20, V_masked) subject maps, A_i: (T_i, 20) time courses
```

Subject maps are back-reconstructed from each subject's own reduced data and then
phase-aligned to the group maps, so `result.subjects` is in input order and every `S_i` is
comparable to `S_group` component by component.

`n_subject` is how many components to keep per subject before concatenation. Setting it
above `n_components` retains subject-specific variance for back-reconstruction to recover.
It cannot exceed a subject's actual rank — see the gotcha below.

## Choosing an estimator

| `estimator` | Method |
|---|---|
| `'cebm'` (default) | Complex ICA by entropy bound minimization. Randomly initialized; pass `rng` for reproducibility. |
| `'nc-fastica'` | Noncircular complex FastICA. Deterministic. |

Note the **hyphen** in `'nc-fastica'`; an underscore raises `ValueError`.

```python
result = run_complex_ica([X], n_components=20, estimator='nc-fastica')
```

## Working with NIfTI files

GIFT stores each complex volume as two ordinary NIfTI files, distinguished by an
underscore-delimited tag. Its defaults use a different tag for reading than for writing:
reading looks for a **suffix** (`_R`/`_I` for real and imaginary, `_CM`/`_CP` for
magnitude and phase) and also accepts the prefix form, while writing emits a **prefix**
(`R_`/`I_`, `Mag_`/`Phase_`; phase in radians). `read_complex` sidesteps this entirely by
taking the two paths explicitly, whatever they are named.

`run_from_files` is the file-level driver: it loads the pairs, runs the pipeline, expands
the maps back to full volume shape, and writes each component out as a complex pair.

```python
from gift.driver import run_from_files

result, written = run_from_files(
    [('R_sub01.nii', 'I_sub01.nii'),
     ('R_sub02.nii', 'I_sub02.nii')],
    n_components=20,
    out_dir='out/',
    estimator='cebm',
)
# written: [(out/R_component_001.nii, out/I_component_001.nii), ...]
```

Pass `complex_type='magnitude&phase'` (and matching `naming=('Mag_', 'Phase_')`) for
magnitude/phase inputs.

To read or write a complex volume yourself, use keyword arguments — `read_complex` infers
the representation from **which** arguments you supply, and supplying half a pair is an
error:

```python
from gift.complex_io import read_complex, write_complex

z = read_complex(real_file='R_sub01.nii', imag_file='I_sub01.nii')   # (*dims, T) complex
z = read_complex(mag_file='Mag_sub01.nii', phase_file='Phase_sub01.nii')
```

If you supply all four, `read_complex` logs a warning and uses the real/imaginary pair,
which reproduces the stored values exactly rather than reconstructing them through
trigonometry.

To get maps back into volume shape, `unmask` undoes the masking, filling excluded voxels
with zero:

```python
from gift.driver import unmask

vols = unmask(result.S_group, result.mask, dims)   # (N, *dims), zero outside the mask
```

## Gotchas

**Component numbering is not stable — across runs, or against the Rust port.** ICA fixes
neither the order nor the phase of its components, and no port canonicalizes them. So
`component_001` here and `component_001` from the Rust driver are generally *different*
sources, even for the deterministic estimator, where the two agree to ~4e-14 once matched
up. **Match components by correlation, never by index or filename.**

**Model order cannot exceed the data's rank.** `whiten_hermitian` raises `ValueError`
rather than returning a NaN-contaminated result. If you hit

    whiten_hermitian: data is rank-deficient at the requested model order ...

lower `n_components` or `n_subject`. Exactly-rank-deficient data (for example, noise-free
simulations) will trip this the moment you ask for more components than the true number of
sources.

**The mask is computed on phase, not magnitude.** Voxels whose phase is untrustworthy are
dropped, because their phase carries noise rather than signal. Pass `mag_mask=` to
intersect a brain/magnitude mask with the phase-quality mask.

**The PDV mask's numeric core is verified against MATLAB; its cleanup step is not.** The
phase-derivative-variance map and the fixed-threshold, AND-over-time, 80%-vote logic are
ported exactly and checked to ~1e-15 against a transcription of the reference. The one
exception is the per-slice morphological opening: `skimage`'s disk differs from MATLAB's
`strel('disk', R)` decomposition, so PDV masks can differ from GIFT by a voxel or two at
region boundaries. Set `erode_radius=0, dilate_radius=0` (via `phase_mask.pdv_mask`) to
drop that step entirely.

## Choosing a phase-quality mask

Two independent measures decide which voxels to keep, selected by `mask_method`:

| `mask_method` | Measures | Needs `dims`? |
|---|---|---|
| `'temporal'` (default) | Per-voxel phase **stability over time**; Otsu-thresholded, one mask from all subjects at once | no |
| `'pdv'` | GIFT's spatial **phase-derivative variance** within each slice; fixed 0.2 threshold, one mask per subject then an 80% vote | yes |

They answer different questions and are not interchangeable. `'temporal'` asks whether a
voxel's phase holds still across timepoints; `'pdv'` asks whether phase is spatially smooth
within a slice, which is the quantity GIFT's Complex toolbox actually computes (ported from
Rodriguez et al. 2011). Use `'pdv'` when you want parity with MATLAB GIFT.

```python
# temporal (default) — no geometry needed
result = run_complex_ica([X], n_components=20, rng=np.random.default_rng(0))

# PDV — needs the (X, Y, S) volume shape
result = run_complex_ica(
    [X], n_components=20, mask_method='pdv', dims=(64, 64, 30),
    rng=np.random.default_rng(0),
)
```

`run_from_files` supplies `dims` automatically from the NIfTI headers, so there you only
pass `mask_method='pdv'`. See the gotcha below on the one PDV step not verified against
MATLAB.

## What the pipeline does

Five stages, in order:

1. **Phase-quality mask**, shared across subjects. The default `'temporal'` method drops
   voxels with time-varying phase (a static per-voxel B0/receiver phase offset provably
   does not affect it); `'pdv'` drops voxels where phase is spatially rough. See "Choosing
   a phase-quality mask" above.
2. **Two-stage complex PCA.** Reduce each subject to `n_subject` components, concatenate
   temporally, then reduce to `n_components` at the group level. For a single subject this
   degenerates to ordinary PCA.
3. **Complex ICA** on the group-reduced data, via the chosen estimator.
4. **Phase correction** of the group maps, which resolves the complex scaling ambiguity
   inherent to complex ICA by fixing a phase per component.
5. **Back-reconstruction** of each subject, then phase alignment to the group maps. The
   subject's mixing is counter-rotated by the inverse phase, so the returned pair still
   satisfies `X_i ≈ A_i @ S_i`.

## Development

The project uses the `giftenv` micromamba environment:

    micromamba run -n giftenv pytest tests -q

Linting and formatting use ruff, configured in `pyproject.toml`:

    micromamba run -n giftenv pipx run ruff check .
    micromamba run -n giftenv pipx run ruff format .

MATLAB oracle fixtures live in `../complex_ica_fixtures/` and are read-only — they were
generated once from MATLAB and pin this port's numerics to the reference implementation.

## License

GPL-3.0-or-later. The complex ICA-EBM and nc-FastICA implementations derive from the
Adali-lab (MLSP/UMBC) MATLAB code.
