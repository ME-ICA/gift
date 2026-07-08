# Complex-Valued ICA in GIFT — Porting Reference

A developer reference for realizing valid complex-valued fMRI ICA in three targets:

- **Python** — a fresh port (NumPy/SciPy).
- **Rust** — a fresh port (`ndarray` + LAPACK, or `nalgebra`).
- **Modified MATLAB** — patching a *working* complex ICA back into GIFT in place
  (wiring up the orphaned dialogs + vendored Adalı-lab algorithms).

It documents (a) what the GIFT codebase actually implements and where, (b) the
complex-fMRI-specific steps that GIFT's generic pipeline omits and must be brought in from
the literature, and (c) target-specific guidance for each of the three (§12).

> **Scope:** single-dataset **complex ICA** (the `R_`/`I_` or `Mag_`/`Phase_` pipeline),
> *not* IVA. See the note on IVA at the end for why IVA is a different model.

---

## 0. TL;DR — state of complex ICA in this GIFT tree

Complex-valued ICA in this version of GIFT is **partially vestigial**. The I/O, complex
PCA/whitening, and complex ICA *option dialogs* exist, but:

- The dedicated complex ICA algorithms (**Complex Infomax**, **Complex Fast ICA**,
  **Complex CMN**) have option dialogs in `icatb_icaOptions.m` but are **not** in the
  algorithm dispatcher `icatb_icaAlgorithm.m` — nothing calls them.
- `dataType` is **hardcoded to `'real'`** in both the GUI setup
  (`icatb_setup_analysis.m:619`) and the batch reader
  (`icatb_read_batch_file.m:220`); there is no exposed, supported switch to turn on
  complex mode.
- `ICA-EBM` and `ERBM`/`FBSS` as shipped are the **real-valued** versions
  (see their headers and the dispatcher comments at `icatb_icaAlgorithm.m:207,214`).
- The only genuinely complex-capable estimator in the tree is **IVA-G**
  (`icatb_iva_second_order.m`) — but that's a different (multi-dataset) model.

**Implication for porting:** mirror GIFT's I/O and complex whitening (they are correct
and complex-safe), but source the *estimator* (complex FastICA / complex ICA-EBM) and
the *phase-specific pre/post-processing* from the complex-fMRI literature — they are
absent or non-functional here.

---

## 1. End-to-end pipeline

| Step | What | In GIFT? | Reference |
|------|------|----------|-----------|
| 1 | Assemble complex volume from two files | ✅ `icatb_loadData.m:79-85` | — |
| 2 | **Phase-quality mask** (quality-map thresholding) | ❌ must add | Rodriguez 2011 |
| 3 | (optional) Remove structured background phase | ❌ must add | Calhoun 2002 |
| 4 | De-mean (choose axis deliberately) | ✅ `icatb_remove_mean.m` | — |
| 5 | Complex whitening (Hermitian, or SUT for non-circular) | ✅ `icatb_pca_whitening.m` | — |
| 6 | **Complex ICA** estimation | ❌ orphaned in tree; use vendored Adalı-lab code | `complex_ica_reference/` |
| 7 | **Phase-ambiguity correction** | ❌ must add | Rodriguez 2012 |
| 7b | (group) Cross-subject phase alignment | ❌ must add | — |

Steps 1, 4, 5 can be copied from GIFT's actual code. Steps 2, 3, 7 come from the
literature. Step 6 is ported from the Adalı-lab reference implementations vendored under
[`complex_ica_reference/`](complex_ica_reference/); see §11 for validating the port
against them.

The pipeline (§2–§9) is **language-agnostic** — the same seven steps apply to all three
targets. §12 gives the target-specific realization (libraries, data types, and — for the
in-place MATLAB patch — the exact GIFT files to change).

---

## 2. Step 1 — Assemble the complex volume

GIFT reads **two ordinary NIfTI/Analyze image files per volume** (image-domain
reconstructed complex data — *not* raw k-space) and combines them
(`icatb_loadData.m:79-85`):

```matlab
if strcmpi(complex_type, 'real&imaginary')
    data = complex(data, data2);                        % real & imaginary
else
    data = complex(data.*cos(data2), data.*sin(data2));  % magnitude & phase → complex
```

### File organization

Two files per image, distinguished by a **prefix/suffix around an underscore**
(`icatb_get_complex_files_naming.m`; filenames **must** contain `_` or it errors at
`:90`). Defaults in `icatb_defaults.m:512-519`:

| Complex type | First part | Second part |
|--------------|-----------|-------------|
| `real&imaginary`  | `R_`   | `I_`     |
| `magnitude&phase` | `Mag_` | `Phase_` |

e.g. `R_sub01_run1.nii` / `I_sub01_run1.nii`. Read and write naming are configured
separately (`READ_NAMING_COMPLEX_IMAGES` / `WRITE_NAMING_COMPLEX_IMAGES`). Outputs are
split back into two files on save (`icatb_calculateICA.m:637-655`).

### Config fields (on `sesInfo.userInput`)

- `dataType` = `'real'` | `'complex'`
- `read_complex_images` / `write_complex_images` = `'real&imaginary'` | `'magnitude&phase'`
- Derived: `read_complex_file_naming` / `write_complex_file_naming`
  (`icatb_name_complex_images.m`)
- Display: `COMPLEX_IMAGES_PER_FIGURE` (`icatb_defaults.m:358`)

---

## 3. Step 2 — Phase-quality mask (quality-map thresholding)

**Not in the GIFT tree — implement from the literature.**

### Why

Phase is trustworthy only where SNR is high. In real brain voxels the complex time
series points in a nearly consistent direction over time (BOLD-induced phase changes are
a small fraction of a radian). In noise voxels (air, CSF, dropout) the phase wanders
randomly. **Temporal phase stability is a proxy for SNR.** Key property: the measure
below is invariant to a constant per-voxel phase offset (B0/receiver background phase),
because it measures *variation* over time — so it can be computed on raw complex data
before any background-phase removal.

### Measure

For voxel `v` with complex time series `z(v,t)`, `t = 1..T`, the magnitude-weighted mean
resultant length (circular concentration):

```
Q(v) = | Σ_t z(v,t) |  /  Σ_t |z(v,t)|          ∈ [0, 1]
```

- Phase stable → samples align → `Q ≈ 1`.
- Phase random → samples cancel → `Q ≈ 0`.

An equivalent operationalization thresholds on **low circular standard deviation of the
phase**, `σ_φ(v)`.

> ⚠️ The *form* (temporal phase consistency) and this resultant-length operationalization
> are what these methods use, but the exact expression/normalization differs between the
> 2011 and 2012 papers. Treat this as the standard template and verify against the paper
> you are matching.

### Mask

```
mask(v) = magnitude_mask(v)  AND  ( Q(v) > τ )
```

- **`magnitude_mask` first**: temporal-mean magnitude above a brain threshold, or a
  standard anatomical/BET brain mask — removes non-brain voxels.
- **`Q > τ`**: "quality-map thresholding" means choosing `τ` **automatically from the
  histogram** of `Q` over in-brain voxels. The distribution is bimodal (noise lobe +
  signal lobe); use an **Otsu threshold** or the valley between modes. A fixed
  `τ ≈ 0.7–0.9` is a manual fallback, but histogram/Otsu selection is the method's actual
  contribution.
- Failing voxels are **dropped** from the ICA matrix (same as any masked voxel), not
  zeroed.

### Pseudocode

```python
Z = complex_volume.reshape(n_voxels, T)          # step 1 output

mag_mean = np.abs(Z).mean(axis=1)
mag_mask = mag_mean > magnitude_threshold        # or an external brain mask

num = np.abs(Z.sum(axis=1))                       # |Σ_t z|
den = np.abs(Z).sum(axis=1) + eps                 # Σ_t |z|
Q   = num / den                                   # phase consistency ∈ [0,1]

tau  = otsu_threshold(Q[mag_mask])                # histogram-derived
mask = mag_mask & (Q > tau)

X = Z[mask, :]                                    # → de-mean → whiten → complex ICA
```

---

## 4. Step 3 — Background-phase removal (optional)

**Not in the GIFT tree.** The acquired phase includes a slowly-varying spatial offset
(B0 inhomogeneity, receiver phase, off-resonance) unrelated to signal. Complex ICA is
only identifiable up to a per-source phase rotation, but the *spatially structured*
background phase is often removed/normalized first so the phase reaching ICA reflects
task/BOLD-related variation. Reference: Calhoun et al. 2002.

---

## 5. Step 4 — De-meaning

GIFT default `preproc_type = 'remove mean per timepoint'` (`icatb_dataReduction.m:67`),
implemented by `icatb_remove_mean.m`, which subtracts the mean of **each column**. With
data shaped `(Voxels × Time)`, a column is a timepoint, so this removes the **spatial
mean of each volume**. Complex-safe (complex mean subtraction is well-defined).

> **Choose the axis deliberately.** Per-volume spatial de-mean (GIFT default) and
> per-voxel temporal de-mean (removes the static complex baseline image) mean different
> things for complex data. Decide based on your model.

### ⚠️ Normalization options are NOT complex-safe

If you port `icatb_preproc_data.m`'s other options verbatim:

- **Intensity normalization** (`:81`): `100 ./ (mean(tmp) + eps)` divides by the
  *complex* temporal mean — meaningless for complex data. Normalize by **magnitude**
  (`abs`), not the complex mean.
- **Variance normalization** (`:90-91`): `std` of complex columns mixes real/imag
  variance. Handle real/imag explicitly or use magnitude.

Safe default (what GIFT uses): **remove-mean only**.

---

## 6. Step 5 — Complex whitening

`icatb_pca_whitening.m` is correctly complex:

```matlab
Xk = Xk - mean(Xk, 2);
[U, d] = eig(Xk*Xk'/(T-1), 'vector');   % Xk' is the CONJUGATE (Hermitian) transpose
...
W_whiten   = U1' ./ s1;
W_dewhiten = U1 .* s1';
```

Using `'` (Hermitian) makes the covariance `E[xxᴴ]`; `eig` returns real eigenvalues; the
whitening is valid for complex data. `icatb_calculate_pca.m` similarly uses complex
random init (`:677`) and SVD (`:694`, "preferred for complex valued data").

### Circular vs non-circular

GIFT whitening uses only the covariance `E[xxᴴ]` → **circular (proper) whitening**.
Complex fMRI sources are generally **non-circular** (unequal/correlated real & imag
parts), so the pseudo-covariance `E[xxᵀ]` carries signal. If your estimator exploits
non-circularity, use the **Strong Uncorrelating Transform (SUT)**, which simultaneously
diagonalizes covariance and pseudo-covariance. GIFT exposed this as the "SUT" option in
the (orphaned) Complex CMN dialog (`icatb_icaOptions.m:975`). Whether you need SUT vs.
plain whitening depends on the estimator.

---

## 7. Step 6 — Complex ICA estimation

**No functional complex ICA estimator in this GIFT tree.** The Complex Infomax / Complex
Fast ICA / Complex CMN dialogs exist (`icatb_icaOptions.m:910-997`) but are not
dispatched. Port from the **Adalı lab (MLSP/UMBC)** originals instead — these are the
authoritative implementations by GIFT's own complex-ICA contributor, vendored into this
repo under [`complex_ica_reference/`](complex_ica_reference/) (GPL v3; see that folder's
`README.md`). Each maps to one of GIFT's orphaned dialogs:

| Reference (in `complex_ica_reference/`) | Algorithm | Entry fn | GIFT orphaned dialog | Paper |
|---|---|---|---|---|
| `complex_EBM/complex_ICA_EBM/complex_ICA_EBM.m` | Complex ICA-EBM | `CEBM(X)` | (successor to Complex Infomax) | Li & Adalı 2010 |
| `PackageCERBM/PackageCodeCERBM/CERBM.m` | Complex ICA-ERBM (adds nonwhiteness + noncircularity) | `CERBM(X,Lite,p)` | — | Fu et al. 2015 |
| `nonCircComplexFastICAsym.m` | Noncircular complex FastICA | `nonCircComplexFastICAsym(X,typeStr)` | Complex Fast ICA (`icatb_icaOptions.m:937`) | Novey & Adalı 2008 |
| `TCMNsym.m` / `ACMNsym.m` | Complex maximization of non-Gaussianity | `doCMNsym(...)` / `ACMNsym(...)` | Complex CMN (`icatb_icaOptions.m:954`) | Novey & Adalı 2008 |

**Recommended first port: Complex ICA-EBM (`CEBM`)** — the most established for complex
fMRI and pairs naturally with GIFT's existing complex whitening. Use `nc-FastICA` if you
want a simpler fixed-point method that exploits **noncircularity** (this is where the
SUT / pseudo-covariance whitening from Step 5 matters); use `CERBM` if you also want to
exploit temporal (sample) dependence.

**Interface conventions** (all reference fns): input `X` is `(N × T)` = sources/channels ×
samples (for fMRI, the PCA-reduced `numComp × voxels`); outputs are `W` (demixing) and
usually `Ahat`, `Shat = W*X`.

> ⚠️ **`CEBM`/`CERBM` depend on precomputed lookup tables** (`nf_table.mat`,
> `complex_nf_table.mat`) that parameterize the entropy-bound nonlinearities. A faithful
> port must export these to a portable format or regenerate them from the paper — the
> algorithm is not self-contained without them. See `complex_ica_reference/README.md`.

---

## 8. Step 7 — Phase-ambiguity correction

**Not in the GIFT tree.** This is the complex-domain analogue of real ICA's sign/scale
normalization.

### The ambiguity

Complex ICA recovers each source up to an arbitrary complex scalar, including an
arbitrary phase rotation `e^{jθ_k}`:

```
X = A·S  ⟺  X = (A·D⁻¹)(D·S),   D = diag(c_k · e^{jθ_k})
```

So the split of a component's energy between its real and imaginary parts is arbitrary,
and components from different runs/subjects aren't comparable until `θ_k` is fixed to a
convention.

### Convention

Rotate each component so **energy is maximally concentrated in the real part** (imag part
minimized). The voxel values `s_k(v)` form an elongated cloud `≈ r(v)·e^{jα}`; the
correction rotates that line onto the real axis.

### Closed form (double-angle / principal-axis trick)

```
θ_k = −½ · angle( Σ_v s_k(v)² )        # sum over MASKED (high-quality) voxels only
s_k ← s_k · e^{ j·θ_k}                  # map: line rotated onto real axis
a_k ← a_k · e^{−j·θ_k}                  # time course absorbs inverse → preserves X = A·S
```

Squaring maps the ±α line ambiguity to a single angle `2α`; after rotation
`Σ_v (s_k·e^{jθ_k})²` is real and positive.

> ⚠️ The sum-of-squares principal-axis formula is the correct "maximize real energy"
> solution and is what this convention amounts to. Whether a given paper uses this exact
> closed form vs. an equivalent optimization/histogram variant differs by paper — verify.

### Residual sign (π) ambiguity

The rotation fixes the axis but leaves a π (direction) ambiguity — the leftover real-ICA
sign ambiguity. Resolve with a fixed criterion (mirror your real-ICA convention):

```
if sign_criterion(Re(s_k)) < 0:   # e.g. sign of skewness, or sum over suprathreshold voxels
    θ_k += π                       # equivalently negate s_k and a_k
```

### Practical notes

1. Compute `θ_k` from **masked (high-quality) voxels only** — noise voxels bias `Σ s²`.
2. Magnitude/scale is a **separate** normalization (z-scoring etc.); phase correction only
   fixes rotation. The two are independent.

### Alternative: histogram-mode method

Build the phase histogram of `angle(s_k(v))` over masked voxels, find the dominant mode
`φ_mode`, set `θ_k = −φ_mode` (Yu et al. 2015 restrict the result to `[−π/2, π/2]`). More
robust to skewed clouds; agrees with the closed form for a clean ellipse.

### Pseudocode (per component)

```python
def correct_phase(s_k, a_k, mask, sign_of=lambda re: np.sign(skew(re))):
    sm = s_k[mask]
    theta = -0.5 * np.angle(np.sum(sm**2))          # orient major axis to real axis
    s_k = s_k * np.exp(1j*theta)
    a_k = a_k * np.exp(-1j*theta)                    # preserve X = A·S
    if sign_of(np.real(s_k[mask])) < 0:              # resolve residual π ambiguity
        s_k, a_k = -s_k, -a_k
    return s_k, a_k
```

---

## 9. Step 7b — Group cross-subject phase alignment

For group complex ICA, correct each subject's component independently (step 7), then
align to a common reference (e.g. the aggregate/group map) so per-subject rotations don't
reintroduce non-physiological variance before group statistics:

```
θ_align = angle( Σ_v s_ref(v)* · s_subj(v) )     # * = complex conjugate
s_subj ← s_subj · e^{−j·θ_align}
```

---

## 10. Note on IVA (why it's not this pipeline)

The only genuinely complex-capable estimator in the tree, **IVA-G**
(`icatb_iva_second_order.m`; auto-detects complex input, handles circular & non-circular
Gaussian cost via gradient/Newton/quasi-Newton; cites Anderson, Li & Adalı 2012), is a
**different model**: it decomposes **K datasets jointly**, linking components across
datasets via source component vectors, and relies on cross-dataset dependence (so it can
even separate Gaussian sources). It is *not* a drop-in for single-dataset complex ICA —
for one complex dataset there is nothing to link. IVA-G is an excellent porting target
**if** your target is group/multi-subject joint decomposition; it is out of scope for the
single-dataset complex ICA pipeline above.

---

## 11. Validating the port (against the reference implementations)

The Adalı-lab MATLAB code vendored in [`complex_ica_reference/`](complex_ica_reference/)
serves as a **correctness oracle**. Recommended verification loop:

1. **Generate ground-truth data.** Run `simulate_complex_fmri_sources.m` (or the demos
   `complex_ICA_EBM/demo1.m`, `PackageCodeCERBM/demo.m`) to produce complex sources +
   mixtures with known `A`, `S`.
2. **Get the gold-standard decomposition.** Run the reference MATLAB (`CEBM`, `CERBM`,
   `nonCircComplexFastICAsym`, …) on that data; save `W`, `Ahat`, `Shat`.
3. **Run your port** on the *same* input matrix.
4. **Compare up to the known indeterminacies.** Complex ICA is exact only up to
   permutation and a per-source complex scale/phase (§8). Match components (e.g. by
   maximum complex correlation), resolve the phase (§8 correction), then check agreement —
   e.g. inter-symbol-interference (ISI) between the true and estimated global matrices, or
   complex correlation between matched sources, at a tight tolerance.
5. **Export the `.mat` lookup tables** (`nf_table.mat`, `complex_nf_table.mat`) to your
   port's format and confirm your nonlinearity values match the MATLAB ones pointwise
   *before* trusting the full decomposition — a table mismatch is the most likely silent
   divergence.

This is the right altitude for verifying a numerical port: agreement with the reference
on known data, not just "it runs" or "the maps look plausible."

---

## 12. Target-specific guidance

The pipeline is the same for all three; what differs is the linear-algebra substrate,
complex-number handling, I/O, and (for MATLAB) integration into GIFT. The vendored
Adalı-lab code (§7) is the common source for the Step 6 estimator in every target.

### 12.0 Shared: export the lookup tables once

`CEBM`/`CERBM` depend on `nf_table.mat` / `complex_nf_table.mat`. Export them **once** to a
language-neutral format up front, so Python and Rust share identical tables and can be
diffed against MATLAB:

```matlab
% in MATLAB, from complex_ica_reference/
t = load('complex_EBM/complex_ICA_EBM/nf_table.mat');
% inspect variables, then write each as CSV / .npy-friendly text
```

Keep the exported tables under version control next to the port so the numbers are pinned.

### 12.1 Python (fresh port)

| Concern | Choice |
|---|---|
| Arrays | NumPy, `dtype=np.complex128` throughout |
| Whitening | `numpy.linalg.eigh` on the **Hermitian** covariance `X @ X.conj().T` (returns real eigenvalues) — mirrors `icatb_pca_whitening.m` |
| Non-circular (SUT) | joint diag of covariance + pseudo-cov `X @ X.T`; `scipy.linalg` |
| NIfTI I/O | `nibabel`; assemble complex from the two files per §2 |
| Otsu threshold (§3) | `skimage.filters.threshold_otsu` |
| Lookup tables | load the §12.0 export (`np.load`) |
| Tests | `pytest`; load reference `W`/`Shat` from `.mat` via `scipy.io.loadmat`, compare per §11 |

- **IVA is free:** if you take the IVA route (§10), `github.com/SSTGroup/independent_vector_analysis`
  is already a maintained Python package — no port needed.
- Watch the transpose/orientation: reference fns use `(N × T)` = channels × samples; NumPy
  default C-order is fine, just be explicit.
- `numpy.angle`, `numpy.exp(1j*θ)` cover the phase-quality (§3) and phase-ambiguity (§8)
  arithmetic directly.

### 12.2 Rust (fresh port)

| Concern | Choice |
|---|---|
| Complex scalars | `num-complex` (`Complex<f64>`) |
| Arrays + linalg | `ndarray` + `ndarray-linalg` (LAPACK backend: OpenBLAS/Netlib) — has `eigh` for Hermitian, `eig`/`svd` for general/complex; or `nalgebra` |
| Whitening | `eigh` on the Hermitian covariance (real eigenvalues); conjugate-transpose via `.t().mapv(Complex::conj)` |
| NIfTI I/O | `nifti` crate; assemble complex per §2 |
| Otsu | small hand-rolled histogram threshold (no heavy dep needed) |
| Lookup tables | `include_bytes!`/`include_str!` the §12.0 export, or load at runtime |
| Tests | `cargo test`; check against reference outputs exported from MATLAB as CSV/npy |

- **This is the heaviest lift.** No complex-ICA crate exists — you port the estimator
  math yourself. Do Python first and treat it as the intermediate oracle; then port
  Python→Rust with numeric parity tests at each step.
- Confirm your LAPACK backend supports the complex (`z`) routines you call (`zheev`,
  `zgeev`, `zgesvd`).
- Be deliberate about conjugate vs. plain transpose — Rust won't catch a `H` vs `T`
  mistake the way a domain-aware reviewer would; the covariance must be Hermitian, the
  pseudo-covariance must be plain-transpose.

### 12.3 Modified MATLAB — patch working complex ICA into GIFT in place

This target does **not** re-port anything: the complex I/O, complex PCA/whitening
(`icatb_pca_whitening.m`), and de-meaning already work. The job is to (a) re-wire the
orphaned dialogs to real algorithms, (b) un-hardcode `dataType`, and (c) add the two
phase steps. Concrete edits:

**a. Drop in the estimators.** Copy the vendored functions into
`GroupICAT/icatb/icatb_analysis_functions/icatb_algorithms/`, renamed to GIFT's
`icatb_` convention (e.g. `icatb_complex_ica_ebm.m` wrapping `CEBM`), and ship their
`.mat` tables alongside. Keep the GPL v3 headers (§License).

**b. Register the algorithms** in `icatb_icaAlgorithm.m`:
- Add names to the `icaAlgo = char(...)` lists (lines **23**, **30**, **35** — three
  modality-specific lists).
- Add dispatch `case`s next to the existing real ones (`case 'ica-ebm'` at **:207**,
  `case {'fbss','erbm'}` at **:214**), e.g.:
  ```matlab
  case 'complex ica-ebm'
      W = icatb_complex_ica_ebm(data);   % data is complex here
      icasig_tmp = W*data;
      A = pinv(W);
  ```
  Note the existing option dialogs already exist in `icatb_icaOptions.m`
  (`complex infomax` **:910**, `complex fast ica` **:937**, `complex cmn` **:954**) — reuse
  or rename them to match your algorithm names.

**c. Un-hardcode `dataType`.** It is forced to `'real'` in two places; expose `'complex'`:
- `icatb_setup_analysis.m:619` (GUI setup)
- `icatb_read_batch_file.m:220` (batch) — also read a `dataType`/`read_complex_images`
  field from the input file so batch scripts can select complex.

**d. Phase-quality mask (Step 2).** Add to mask creation
(`icatb_helper_functions/icatb_createMask.m`, and `icatb_parallel_files/icatb_parCreateMask.m`)
a complex branch computing `Q(v)` (§3) and intersecting it with the magnitude mask when
`dataType` is `'complex'`.

**e. Phase-ambiguity correction (Step 7).** Apply the §8 rotation in post-processing,
where components are finalized — `icatb_calculateICA.m` around the complex-write block
(**:637-655**), or in `icatb_scaleICA.m` / `icatb_calibrateComponents.m` (the existing
sign/scale normalization stage). Rotate `icasig` and apply the inverse to `A`.

**f. Validate** against the standalone reference (§11): run GIFT-complex and `CEBM` on the
same reduced matrix and confirm they agree up to permutation/phase.

> **Effort ordering:** b→c gets the *estimator* running end-to-end in GIFT (biggest win,
> smallest change, since I/O + whitening already work). d and e add the complex-fMRI
> correctness that the generic pipeline lacks. Do b/c first, verify, then layer d/e.

## License

The complex ICA algorithms this port is based on (Adalı lab, MLSP/UMBC) are **GPL v3**.
Porting them line-for-line produces a derivative work, so **this port is GPL v3**. The
vendored originals under `complex_ica_reference/` retain their GPL v3 headers; cite the
papers (see that folder's `README.md`) in any related work.

---

## References

- Calhoun, Adalı, Pearlson, van Zijl, Pekar (2002). *Independent component analysis of
  fMRI data in the complex domain.* Magn. Reson. Med. 48(1):180–192.
- Rodriguez, Correa, Eichele, Calhoun, Adalı (2011). *Quality map thresholding for
  de-noising of complex-valued fMRI data and its application to ICA of fMRI.* J. Signal
  Process. Syst. 65:497–508.
- Rodriguez, Calhoun, Adalı (2012). *De-noising, phase ambiguity correction and
  visualization techniques for complex-valued fMRI data.* Pattern Recognition
  45:2050–2063.
- Yu, Lin, Kuang, Gong, Cong, Calhoun (2015). *ICA of full complex-valued fMRI data using
  phase information of spatial maps.* J. Neurosci. Methods.
- Bingham & Hyvärinen (2000). *A fast fixed-point algorithm for independent component
  analysis of complex valued signals.* Int. J. Neural Syst. 10(1):1–8.
- Novey & Adalı (2008). *On extending the complex FastICA algorithm to noncircular
  sources.* IEEE Trans. Signal Process. 56(5):2148–2154. (nc-FastICA.)
- Novey & Adalı (2008). *Complex ICA by negentropy maximization.* IEEE Trans. Neural
  Netw. 19(4):596–609. (T-CMN / CMN.)
- Li & Adalı (2010). *Complex independent component analysis by entropy bound
  minimization.* IEEE Trans. Circuits Syst. I, 57(7):1417–1430. (Complex ICA-EBM.)
- Fu, Phlypo, Anderson, Adalı (2015). *Complex independent component analysis using three
  types of diversity: non-Gaussianity, nonwhiteness, and noncircularity.* IEEE Trans.
  Signal Process. 63(3):794–805. (Complex ICA-ERBM / CERBM.)
- Anderson, Li, Adalı (2012). *Complex-valued independent vector analysis: Application to
  multivariate Gaussian model.* Signal Process. 92:1821–1831. (IVA, for context.)

Reference implementations (Adalı lab, MLSP/UMBC, GPL v3) are vendored under
[`complex_ica_reference/`](complex_ica_reference/); see its `README.md`.

---

*Code references (`file:line`) are to this GIFT tree and should be re-verified against the
version you port from; they may drift across releases.*
