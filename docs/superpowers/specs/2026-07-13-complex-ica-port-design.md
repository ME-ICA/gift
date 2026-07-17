# Complex-Valued ICA: GIFT Implementation + Python & Rust Ports — Design

**Date:** 2026-07-13
**Status:** Approved design; ready for implementation planning (Phase 0+1 first).

Companion reference (detailed algorithm/file-level notes):
[`doc/complex_ica_porting_reference.md`](../../../doc/complex_ica_porting_reference.md).
Vendored Adalí-lab reference implementations (GPL v3, correctness oracles):
[`doc/complex_ica_reference/`](../../../doc/complex_ica_reference/).

---

## 1. Goal

Deliver valid complex-valued fMRI ICA in three targets:

1. **MATLAB-in-GIFT** — patch a *working* complex ICA into GIFT in place.
2. **Python** — a clean-room port (`complex-gift` package).
3. **Rust** — a port from the validated Python.

Each is independently shippable. All three are **GPL v3** (derivative of the Adalí-lab
complex ICA algorithms).

## 2. Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Port scope** | Complex ICA + *minimal runnable GIFT* | Ports run a full complex **group** analysis end-to-end (data reduction → complex ICA → back-recon → NIfTI outputs). **Excluded:** GUI, dFNC/dFC, MANCOVA, display/rendering, real-valued ICA. |
| **Estimators** | **CEBM + nc-FastICA**, behind a swappable interface | Two independent estimators cross-check the pipeline. (Both whiten internally; see the correction under §4 — neither uses the SUT.) |
| **Sequencing** | **MATLAB → Python → Rust** | Each stage is the numeric oracle for the next; lowest risk. |
| **Phase-step validation** | Property-based + synthetic-injection, identical assertions in all three targets | ~~The phase steps have no reference implementation to diff against; prove properties instead.~~ **Premise corrected — see the second correction under §4.** The Complex GIFT toolbox (linked from the repo root README) *does* ship reference implementations of both phase steps; they were simply not consulted at spec time. |
| **Group scheme** | GIFT default: temporal concatenation → two-stage (subject then group) complex PCA → GICA back-reconstruction | Keeps "minimal runnable" faithful to GIFT without pulling in other back-recon variants. |

## 3. Architecture — one roadmap, three sub-projects

Track 1 (MATLAB) is both a **deliverable** and the **oracle generator**: it exports the
lookup tables and produces the reference decompositions that Tracks 2 and 3 test against.

```
                 ┌─────────────────────────────────────────────┐
                 │  Shared artifacts (produced once, by MATLAB) │
                 │  • nf_table / complex_nf_table  → CSV/npy    │
                 │  • ground-truth fixtures A,S,X (simulated)   │
                 │  • oracle outputs W,A,S from CEBM/ncFastICA  │
                 └─────────────────────────────────────────────┘
                          │ consumed by ↓        │ consumed by ↓
 Track 1: MATLAB-in-GIFT ─┘        Track 2: Python ─┘   Track 3: Rust
 (patch complex ICA into GIFT)     (clean port)         (port from Python)
        │ oracle for ──────────────────▶│ oracle for ─────────▶│
```

Each track gets its own implementation plan; this spec is the umbrella defining the shared
contract so the three stay numerically aligned.

## 4. Shared pipeline decomposition (language-agnostic units)

Same bounded units in all three languages, so tests transfer. Each has one job and a
defined interface.

| Unit | Input → Output | Notes |
|---|---|---|
| `complex_io` | 2 files/volume → complex 4-D array | R_/I_, Mag_/Phase_ (ref §2) |
| `phase_mask` | complex `(V×T)` → boolean mask | temporal quality map `Q(v)` + Otsu (default); **plus** a faithful port of GIFT's spatial PDV mask — see 2nd correction below |
| `preproc` | complex `(V×T)` → de-meaned `(V×T)` | remove-mean only; magnitude-safe norms (ref §5) |
| `whiten` | `(V×T)` → reduced `(N×T)`, W_wh, W_dw | Hermitian **or** SUT (ref §6), selected by estimator |
| `estimator` | `(N×T)` → W, A, S | **swappable**: `cebm` \| `nc_fastica` (ref §7) |
| `phase_correct` | S, A, mask → rotated S, A | closed-form θ + π-sign; verified **equivalent** to GIFT's `icatb_PCA_phase_correction.m` — see 2nd correction below |
| `group` | per-subject reduced data → group W, back-recon | two-stage complex PCA + GICA back-recon + align (ref §7b, §9) |
| `nf_table` | — | loaded from shared CSV/npy export (ref §12.0) |

**Correction (established during Phase 2 — the original claim here was wrong):** this spec
originally stated that the estimator selector picks the whitening path, with
`nc_fastica` ⇒ SUT whitening. That is **not** what the reference implementations do. Both
estimators perform their **own internal** whitening: `cebm` whitens via `pre_processing`
(Hermitian covariance) and separately forms the pseudo-covariance itself; `nc_fastica`
likewise whitens with the Hermitian covariance and carries an explicit pseudo-covariance
term in its fixed-point update. Neither calls the Strong Uncorrelating Transform.

So the units are in fact **fully decoupled**: `whiten` is used by the *group reduction*
(`two_stage_pca`), not by the estimators. `strong_uncorrelating_transform` is retained as a
correct, tested, degeneracy-safe public utility for noncircular work, but **no estimator
currently calls it** — do not assume it is on the hot path.
Everything else is a clean data hand-off.

**Second correction (established post-Phase-3 — the "no reference implementation" premise
was wrong):** the decision table above claimed the phase steps had no reference to diff
against, so both were validated by properties alone. That premise was false. The **Complex
GIFT** toolbox (`GroupICATv2.0d_complex`, linked from the repo-root README all along)
ships MATLAB implementations of both, authored by Pedro A. Rodriguez:

- **Phase-quality mask** — `icatb_preproc_complex_data.m` → `icatb_qmpd.m` →
  `icatb_val_qual_circular.m` → `icatb_qmpd_quality_mask_run.m`. This is a **spatial
  phase-derivative variance (PDV)** map (Rodriguez et al. 2011): local variance of wrapped
  phase gradients within each 2-D slice, keep-low at a fixed 0.2 threshold, ANDed over
  time, morphologically opened per slice, then combined across subjects by an 80% vote.
  This is **not** the algorithm originally ported. `phase_mask.py` shipped a *temporal*
  coherence measure (`|Σ Zₜ| / Σ|Zₜ|` + Otsu), derived from the paper's description rather
  than its code — a different physical quantity. **Resolution:** both are now offered via
  `phase_quality_mask(..., method=)`. `'temporal'` remains the default; `'pdv'` is a
  faithful port, verified to ~1e-15 against a literal transcription of the MATLAB. The one
  unverified step is the per-slice morphological opening (`skimage` disk vs. MATLAB
  `strel('disk')`), which is documented and disable-able.
- **Phase-ambiguity correction** — `icatb_PCA_phase_correction.m`. Its rotation
  (`-atan2(COEFF(2,1), COEFF(1,1))` from an SVD of `[real, imag]`) is **mathematically
  equivalent** to the port's closed form `-0.5·angle(Σ s²)`; verified to ~1e-15. The only
  divergence is the 180° flip criterion — GIFT uses raw `sum(x³)`, the port uses
  mean-centered `scipy.stats.skew`. These are provably sign-identical for zero-mean input,
  and the sources reaching `correct_phase` are zero-mean to ~1e-17 (whitening centers over
  voxels), so the difference is **latent** in the pipeline, reachable only by calling
  `correct_phase` directly on non-centered maps.

Two smaller notes from the same comparison: (1) `icatb_complex_ICA_EBM.m`'s
`pre_processing` uses **Hermitian** whitening (`inv_sqrtmH(X*X'/T)`), confirming the first
correction above — no SUT. (2) GIFT's complex file naming differs between read and write
(`READ_NAMING_COMPLEX_IMAGES` is a *suffix*, `_R`/`_I`, `_CM`/`_CP`; `WRITE_` is a
*prefix*, `R_`/`I_`, `Mag_`/`Phase_`); the port's `read_complex` takes explicit paths and
is naming-agnostic, and its docstrings have been corrected to state the read/write split.

## 5. Track 1 — MATLAB-in-GIFT (the oracle)

No re-porting: complex I/O, complex PCA/whitening, and de-meaning already work. Six edits
(detail in ref §12.3), in effort order:

1. **Vendor estimators** into `icatb_algorithms/` as `icatb_complex_ica_ebm.m` (wraps
   `CEBM`) and `icatb_complex_nc_fastica.m` (wraps `nonCircComplexFastICAsym`), shipping
   their `.mat` tables. Keep GPL v3 headers.
2. **Register** in `icatb_icaAlgorithm.m` — add to the three `icaAlgo` char lists
   (:23/:30/:35) and add dispatch `case`s beside the real ones (:207/:214).
3. **Un-hardcode `dataType`** — expose `'complex'` at `icatb_setup_analysis.m:619` and
   `icatb_read_batch_file.m:220` (batch reads a `dataType`/`read_complex_images` field).
4. **Phase-quality mask** — complex branch in `icatb_createMask.m` +
   `icatb_parCreateMask.m` computing `Q(v)`, intersected with the magnitude mask.
5. **Phase-ambiguity correction** — apply the ref-§8 rotation in `icatb_calculateICA.m`
   (:637–655) / `icatb_scaleICA.m`; rotate `icasig`, inverse-rotate `A`.
6. **Oracle export harness** — script dumping `nf_table`/`complex_nf_table` to CSV/npy and
   writing fixture decompositions (W/A/S on simulated input) to a `fixtures/` dir the other
   tracks read.

Effort ordering: 1→2→3 gets the estimator running end-to-end; 4→5 add complex-fMRI
correctness; 6 unlocks the other tracks.

## 6. Track 2 — Python (`complex-gift`)

One module per unit from §4 (`complex_io.py`, `phase_mask.py`, `whiten.py`,
`estimators/cebm.py`, `estimators/nc_fastica.py`, `phase_correct.py`, `group.py`).

| Concern | Choice |
|---|---|
| Arrays | NumPy `complex128` throughout |
| Whitening | `numpy.linalg.eigh` on Hermitian cov; SUT via joint-diag of cov + pseudo-cov |
| Estimator iface | ABC `Estimator.fit(X) -> (W, A, S)`; `cebm`, `nc_fastica` |
| NIfTI I/O | `nibabel` |
| Otsu | `skimage.filters.threshold_otsu` |
| nf_table | load the Track-1 CSV/npy export |
| Tests | `pytest`; oracle via `scipy.io.loadmat` on Track-1 fixtures |

Milestone order (units bottom-up): `nf_table` diff → `whiten` → `estimators` (oracle-diff)
→ `phase_mask`/`phase_correct` (property) → `group` → end-to-end on fixtures.

## 7. Track 3 — Rust

Heaviest lift — no complex-ICA crate exists. Port **from the finished, validated Python**,
module-for-module, with numeric-parity tests at each step (Python is the intermediate
oracle).

| Concern | Choice |
|---|---|
| Complex | `num-complex` `Complex<f64>` |
| Linalg | `ndarray` + `ndarray-linalg` (LAPACK: `zheev`/`zgeev`/`zgesvd`) |
| Whitening | `eigh` Hermitian; conj-transpose explicit — be deliberate about H vs T |
| nf_table | `include_str!` the shared export |
| Tests | `cargo test` vs fixtures exported as CSV/npy from Python & MATLAB |

Crate modules mirror the Python modules 1:1 so parity tests map directly.

## 8. Validation architecture (unified across all three)

Two mechanisms, because the pipeline has two kinds of units:

- **Oracle-diff** (estimator + whitening + nf_table — these have a ground truth): generate
  simulated `A,S,X` → run Track-1 MATLAB `CEBM`/`nc-FastICA` → diff each port's output up
  to permutation + per-source complex phase/scale (match by max complex correlation, then
  ISI at tight tolerance). Diff the `nf_table` export **pointwise first** — the most likely
  silent divergence.
- **Property / synthetic-injection** (phase steps): identical assertions in all three
  languages. (Originally justified by "no ground truth"; that was wrong — see the second
  correction under §4. Complex GIFT provides a reference, and the Python port now diffs
  the PDV mask and phase correction against transcriptions of it in addition to the
  property tests below.) —
  - `phase_mask`: invariant to a constant per-voxel phase offset, `mask(z) == mask(z·e^{jφ})`
  - `phase_correct`: inject known rotation θ, assert recovery of −θ; assert `‖X − A·S‖`
    unchanged
  - `group` align: known per-subject rotations collapse to zero
- **Shared fixtures:** one version-controlled `fixtures/` dir (simulated inputs + MATLAB
  oracle outputs + exported tables), consumed identically by Python and Rust so the three
  ports provably test the same numbers.

## 9. Roadmap & deliverables

**Phase 0 — Shared foundation (blocks everything)**
- Track-1 edits #1–2 (vendor + register estimators) so GIFT runs complex CEBM/nc-FastICA
  end-to-end.
- Track-1 edit #6: export `nf_table`/`complex_nf_table` → CSV/npy; generate simulated
  fixtures + oracle decompositions.
- **Exit:** `fixtures/` populated + committed; `nf_table` export sanity-checked.

**Phase 1 — MATLAB deliverable complete**
- Track-1 edits #3–5 (un-hardcode `dataType`, phase-quality mask, phase-ambiguity
  correction); phase-step property tests run in MATLAB.
- **Exit:** a real complex group analysis runs inside GIFT from batch; phase-step
  properties pass.

**Phase 2 — Python deliverable complete**
- Build `complex-gift` bottom-up (§6 order).
- **Exit:** Python matches MATLAB oracle within tolerance; property tests pass; end-to-end
  group run reproduces MATLAB outputs.

**Phase 3 — Rust deliverable complete**
- Port module-for-module from validated Python with parity tests per module.
- **Exit:** Rust matches Python (and MATLAB) on shared fixtures; property tests pass.

**Deliverables:** (1) patched GIFT with complex ICA + exported tables/fixtures;
(2) `complex-gift` Python package; (3) Rust crate. Each phase independently shippable.

**Recommended first plan:** Phase 0+1 (MATLAB), since Phases 2–3 cannot begin until the
fixtures exist.

## 10. License

All three targets are **GPL v3** — derivative of the Adalí-lab (MLSP/UMBC) complex ICA
algorithms. Vendored originals retain their GPL v3 headers; cite the papers listed in
`doc/complex_ica_reference/README.md` in any related work.
