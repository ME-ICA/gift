# Proposed Single-Run ICA Features for Python and Rust

## Purpose and scope

This document summarizes candidate GIFT features for ports that operate on one
subject/session data set and perform one ICA execution. In this document, "single-run"
means a single input data matrix or 4-D image series, not group ICA and not independent
vector analysis (IVA).

The proposed scope includes features 1-5 and 7-10 from the reduced candidate list:

1. Real-valued ICA pipeline
2. Core ICA estimators
3. Model-order estimation
4. Efficient PCA methods
5. Reference-guided ICA
7. Component normalization and ambiguity correction
8. Component quality control and labeling
9. Time-course preprocessing
10. Per-run component diagnostics

Feature 6 (spatial and temporal ICA modes) is not treated as a separate work package.
Spatial ICA is the default orientation assumed throughout this proposal. A future temporal
ICA mode can reuse the same low-level PCA and estimator interfaces after its array and
output conventions are specified.

Explicit non-goals are:

- temporal concatenation across subjects;
- group-level PCA;
- GICA/GICA2/GICA3 back-reconstruction or dual regression;
- subject alignment and group statistics;
- IVA and cross-dataset source dependence;
- ICASSO/MST or other repeated-run stability analysis;
- MANCOVAN, source-based morphometry, and cohort-level dFNC statistics;
- replication of MATLAB GUIs.

The current Python and Rust packages are useful foundations, but their top-level pipelines
are group-complex-ICA pipelines. Python exposes a common estimator result containing
`W`, `A`, and `S` in
[`python/src/gift/estimators/base.py`](python/src/gift/estimators/base.py#L7-L16),
while the Rust equivalent is in
[`rust/src/estimators/mod.rs`](rust/src/estimators/mod.rs#L7-L13). The proposed real-valued
pipeline should preserve that contract while introducing a single-dataset entry point
instead of extending the existing group entry points in place.

## Shared data and result contract

The numerical API should use one consistent orientation:

- input `X`: `(n_features, n_samples)`;
- demixing matrix `W`: `(n_components, n_features)`;
- mixing matrix `A`: `(n_features, n_components)`;
- sources `S = W @ X_centered`: `(n_components, n_samples)`.

For spatial fMRI ICA, features are time points and samples are masked voxels. The file I/O
layer may naturally load a 4-D NIfTI as `(time, voxel)`, but it should transpose explicitly
at the numerical boundary and record the orientation in metadata.

A proposed result object is:

```text
ICAResult
  W, A, S
  mean
  mask
  pca: eigenvalues, whitening, dewhitening, retained variance
  preprocessing: ordered operations and parameters
  estimator: name, parameters, convergence status, iterations, seed
  normalization: sign/scale transformations
  image: spatial shape and affine, when input came from an image
```

Python should own high-level NIfTI handling, configuration, and diagnostic tables. Rust
should expose array-first numerical functions with typed parameter structures and return
explicit errors for rank deficiency, non-convergence, and invalid parameter combinations.

## 1. Real-valued ICA pipeline

### MATLAB behavior to preserve

GIFT already has a real/complex data-loading boundary. `icatb_loadData` defaults to real
data and accepts `dataType = 'real'` or `'complex'`; the real branch delegates to the
standard image reader
([`icatb_loadData.m`](GroupICAT/icatb/icatb_io_data_functions/icatb_loadData.m#L1-L52)).
The port should retain this separation, but real-valued input should not be routed through
the two-file complex-volume conventions.

The single-run pipeline should contain these stages:

1. Load a numeric matrix or a 4-D NIfTI image series.
2. Construct or load a mask.
3. Apply one configured data preprocessing method.
4. Estimate or validate the model order.
5. Reduce and whiten with PCA.
6. Run one selected ICA estimator.
7. resolve sign/scale ambiguity without changing the reconstruction;
8. expand spatial sources to image space and write maps, time courses, and metadata.

The default GIFT mask includes voxels whose values exceed a per-image mean and combines
valid voxels across the input. The implementation is in
[`icatb_createMask.m`](GroupICAT/icatb/icatb_helper_functions/icatb_createMask.m#L1-L14)
and its default-mask branch begins near
[`icatb_createMask.m`](GroupICAT/icatb/icatb_helper_functions/icatb_createMask.m#L62-L103).
For a single run, the port should support:

- an explicit binary mask;
- a GIFT-compatible mean-threshold mask;
- a finite/nonzero mask suitable for already-preprocessed arrays.

The global `DEFAULT_MASK_OPTION = 'first_file'` is group-oriented and should not be copied
literally
([`icatb_defaults.m`](GroupICAT/icatb/icatb_defaults.m#L487-L496)). For one 4-D run, the
API should state whether mask estimation uses the first volume, a mean image, or all
volumes. A mean-image default is less surprising, while a `gift_first_volume` compatibility
mode can preserve MATLAB parity.

GIFT's preprocessing function accepts the following values
([`icatb_preproc_data.m`](GroupICAT/icatb/icatb_analysis_functions/icatb_preproc_data.m#L1-L18),
[`icatb_preproc_data.m`](GroupICAT/icatb/icatb_analysis_functions/icatb_preproc_data.m#L48-L89)):

- `remove mean per timepoint` (`rt`);
- `remove mean per voxel` (`rv`);
- `intensity normalization` (`in`);
- `variance normalization` (`vn`);
- `none`.

These operations should be ported as independent functions and composed by configuration.
The default numerical type should be `float64`. Optional `float32` input/output may be
supported for large data, but PCA accumulation and convergence diagnostics should remain
in `float64` unless parity tests establish safe tolerances.

### Proposed interfaces

Python:

```python
run_ica(
    data,
    n_components=None,
    mask=None,
    mask_method="mean_image",
    preprocessing="variance_normalization",
    pca="standard",
    estimator="ica_ebm",
    normalization="skewness",
    random_state=None,
) -> ICAResult
```

Rust should mirror this with `RunIcaOptions`, `MaskMethod`, `Preprocessing`, `PcaMethod`,
and `EstimatorKind` enums rather than stringly typed parameters.

### Validation requirements

- Matrix and NIfTI inputs must produce identical masked matrices.
- `A @ S` must reproduce the centered, PCA-retained data within a documented tolerance.
- Explicit-mask and automatic-mask behavior must be tested separately.
- Preprocessing functions require small, hand-computable tests and MATLAB-oracle fixtures.
- Output NIfTI maps must preserve spatial dimensions and affine metadata.

## 2. Core ICA estimators

### Recommended estimator order

The initial estimator set should be deliberately smaller than GIFT's full algorithm
registry. GIFT dispatches many historical algorithms from
[`icatb_icaAlgorithm.m`](GroupICAT/icatb/icatb_analysis_functions/icatb_icaAlgorithm.m#L23-L38).
For a maintainable single-run port, the recommended order is:

1. real-valued ICA-EBM;
2. symmetric and deflationary FastICA;
3. Infomax and Extended Infomax.

ICA-EBM is the most direct extension of the existing complex EBM work and provides flexible
density matching. GIFT dispatches it as `W = icatb_ica_ebm(data)` and derives
`S = W * data`, `A = pinv(W)`
([`icatb_icaAlgorithm.m`](GroupICAT/icatb/icatb_analysis_functions/icatb_icaAlgorithm.m#L209-L214)).
The port should retain a common estimator return type, add convergence information, and
avoid silently replacing a failed inverse with a pseudoinverse without reporting rank.

FastICA is dispatched through `icatb_fastICA`
([`icatb_icaAlgorithm.m`](GroupICAT/icatb/icatb_analysis_functions/icatb_icaAlgorithm.m#L124-L127)).
The configurable MATLAB parameters are defined in
[`icatb_icaOptions.m`](GroupICAT/icatb/icatb_analysis_functions/icatb_icaOptions.m#L208-L315):

| Parameter | GIFT default/options | Proposed meaning |
|---|---|---|
| `epsilon` | `1e-4` | convergence tolerance |
| `maxNumIterations` | `1000` | main iteration limit |
| `maxFinetune` | `5` | fine-tuning iteration limit |
| `sampleSize` | `1` | sample fraction |
| `approach` | `defl` or `symm` | deflationary or symmetric estimation |
| `g` | `pow3`, `tanh`, `gauss`, `skew` | primary nonlinearity |
| `finetune` | `off` or a supported nonlinearity | fine-tuning nonlinearity |
| `stabilization` | `off` or `on` | stabilized updates |

The first implementation should support `approach`, `g`, `epsilon`, and
`max_iterations`. Fine-tuning, subsampling, and stabilization can follow after the core
algorithm matches MATLAB or a second trusted implementation.

Infomax dispatches through `icatb_runica`, combines learned weights with the sphering
matrix, and recomputes sources from the final `W`
([`icatb_icaAlgorithm.m`](GroupICAT/icatb/icatb_analysis_functions/icatb_icaAlgorithm.m#L97-L122)).
Its GIFT options are defined in
[`icatb_icaOptions.m`](GroupICAT/icatb/icatb_analysis_functions/icatb_icaOptions.m#L56-L204):

| Parameter | GIFT default |
|---|---:|
| `block` | `floor(sqrt(frames / 3))` |
| `stop` | `1e-6` |
| `lrate` | `0.015 / log(n_components)` |
| `maxsteps` | `512` |
| `anneal` | `0.9` |
| `annealdeg` | `60` |
| `momentum` | `0` |
| `extended` | `0` |
| `sphering` | `on` |
| `bias` | `on` |

`extended=1` enables the Extended Infomax behavior. Random initialization must accept an
explicit seed, and the actual seed plus termination reason must be stored in `ICAResult`.

### Target allocation

- Python: implement all three estimators behind the existing estimator protocol.
- Rust: prioritize ICA-EBM and FastICA; port Infomax after the shared whitening and
  convergence contracts are stable.
- Both: share fixture inputs and compare outputs after component assignment and per-source
  sign/scale normalization rather than comparing raw component order.

### Validation requirements

- Compare `W`, `A`, and `S` to fixed MATLAB fixtures up to permutation and sign/scale.
- Report reconstruction error, pairwise source correlation, and an interference metric.
- Exercise super-Gaussian, sub-Gaussian, mixed-density, rank-deficient, and nearly Gaussian
  synthetic cases.
- Verify reproducibility for fixed seeds and variability for different seeds.
- Treat non-convergence as structured output or an error, never as silent success.

## 3. Model-order estimation

### MATLAB behavior and parameters

GIFT exposes a single entry point returning the estimated order plus MDL, AIC, and KIC
curves
([`icatb_estimate_dimension.m`](GroupICAT/icatb/icatb_estimate_dimension.m#L1-L24)).
The dimensionality configuration is:

| `method` | Meaning |
|---:|---|
| `1` | information-theoretic criteria with effectively IID sampling |
| `2` | smoothness-corrected estimation using an FWHM vector |
| `3` | entropy-rate estimator with a finite-memory model |
| `4` | entropy-rate estimator with an autoregressive model |

The defaults are `method = 1` and `fwhm = [5, 5, 5]`
([`icatb_defaults.m`](GroupICAT/icatb/icatb_defaults.m#L765-L775)). Method 2 requires an
explicit FWHM setting
([`icatb_estimate_dimension.m`](GroupICAT/icatb/icatb_estimate_dimension.m#L61-L92)).
Methods 3 and 4 branch to the two entropy-rate estimators
([`icatb_estimate_dimension.m`](GroupICAT/icatb/icatb_estimate_dimension.m#L181-L198)).

For the information-theoretic path, GIFT variance-normalizes data, estimates the spectrum,
and constructs AIC, KIC, and MDL curves. The selected result is the first local minimum of
MDL
([`icatb_estimate_dimension.m`](GroupICAT/icatb/icatb_estimate_dimension.m#L340-L376)).

### Proposed scope

The first port should implement method 1 and return all diagnostic curves rather than only
the selected order. Method 2 should follow once image smoothness and voxel-size conventions
are specified. Methods 3 and 4 are scientifically useful but contain a larger collection of
embedded signal-processing routines and should be separate milestones.

Proposed API:

```python
estimate_model_order(
    data,
    mask=None,
    method="iid_mdl",
    fwhm=None,
    precision="float64",
) -> ModelOrderResult
```

`ModelOrderResult` should include the selected order, MDL/AIC/KIC arrays, eigenvalues,
method, effective sample count, and any warnings about conditioning or boundary minima.

### Validation requirements

- Use MATLAB-oracle fixtures for each supported method.
- Test simulated mixtures with known latent rank across multiple SNRs and sample counts.
- Verify behavior when the optimum is at the search boundary or when eigenvalues approach
  machine precision.
- Never present the estimate as a guarantee; expose curves so users can inspect ambiguity.

## 4. Efficient PCA methods

### MATLAB behavior and parameters

`icatb_calculate_pca` is already a largely self-contained numerical API. It accepts numeric
arrays or NIfTI/MAT files and supports `standard`/`evd`, `svd`, `empca`, `mpowit`, `stp`,
and an automatic `best` selection
([`icatb_calculate_pca.m`](GroupICAT/icatb/icatb_analysis_functions/icatb_calculate_pca.m#L1-L48)).
It can also apply a mask, preprocess data, whiten, select a MAT variable, and choose a
method from an available-RAM estimate
([`icatb_calculate_pca.m`](GroupICAT/icatb/icatb_analysis_functions/icatb_calculate_pca.m#L67-L109),
[`icatb_calculate_pca.m`](GroupICAT/icatb/icatb_analysis_functions/icatb_calculate_pca.m#L186-L227)).

Method-specific parameters are defined in
[`icatb_pca_options.m`](GroupICAT/icatb/icatb_helper_functions/icatb_pca_options.m#L1-L235):

| Method | Important parameters and defaults |
|---|---|
| Standard EVD | `stack_data`, `storage={full,packed}`, `precision`, `eig_solver={selective,all}` |
| SVD | `precision`, `solver={selective,all}` |
| EM-PCA | `stack_data`, `precision`, `tolerance=1e-4`, `max_iter=1000` |
| MPOWIT | `stack_data`, `precision`, `tolerance=1e-5`, `max_iter=1000`, `block_multiplier=10` |
| STP | `precision`, `num_comp=500`, `numGroups=10` |

GIFT switches unstacked EM-PCA to MPOWIT
([`icatb_calculate_pca.m`](GroupICAT/icatb/icatb_analysis_functions/icatb_calculate_pca.m#L248-L264))
and floors iterative tolerances at `1e-4` for single-precision data
([`icatb_calculate_pca.m`](GroupICAT/icatb/icatb_analysis_functions/icatb_calculate_pca.m#L296-L319)).
Whitening and dewhitening are derived from the retained eigenvectors and eigenvalues
([`icatb_calculate_pca.m`](GroupICAT/icatb/icatb_analysis_functions/icatb_calculate_pca.m#L1468-L1482)).

The current Python and Rust ports already contain rank-checked Hermitian whitening for
complex data
([`python/src/gift/whiten.py`](python/src/gift/whiten.py#L13-L48),
[`rust/src/whiten.rs`](rust/src/whiten.rs#L14-L67)). The real implementation should reuse
the same failure policy: reject a requested model order whose smallest retained eigenvalue
is not comfortably positive relative to the largest.

### Proposed implementation order

1. Dense EVD and SVD with whitening/dewhitening.
2. Chunked covariance accumulation for file-backed arrays.
3. MPOWIT for large matrices.
4. EM-PCA.
5. STP only if single-run time/voxel dimensions demonstrate a real performance advantage.

Although STP was introduced for large group analyses, its streaming/subsampling ideas may
still benefit one very large 4-D run. It should not be ported merely for feature parity;
benchmarks should determine whether MPOWIT or randomized/truncated SVD already covers the
use case.

### Reproducibility and validation

GIFT can force deterministic PCA initialization with `NORAND_DETERMINISTIC`
([`icatb_calculate_pca.m`](GroupICAT/icatb/icatb_analysis_functions/icatb_calculate_pca.m#L51-L63),
[`icatb_defaults.m`](GroupICAT/icatb/icatb_defaults.m#L777-L781)). Ports should use an
explicit RNG argument instead of a global switch.

Tests should compare:

- retained eigenvalues and subspaces;
- whitening covariance against the identity;
- `dewhitening @ whitened` reconstruction error;
- dense versus streamed results;
- float32 versus float64 error;
- deterministic iterative results for a fixed seed.

## 5. Reference-guided ICA

### Applicable single-run methods

Three GIFT families can be adapted to a single data set with external references:

1. semi-blind Infomax;
2. spatially constrained ICA (fixed-point ICA-R);
3. adaptive-reverse constrained ICA-EBM.

MOO-ICAR/GIG-ICA is excluded from the initial scope because its usual workflow obtains
aggregate maps from a previous group analysis. A future API may accept the same mathematics
when the references are supplied independently, but it should not imply a group-analysis
dependency.

Semi-blind Infomax adds a correlation correction factor `prefs=0.5` to the main Infomax
options
([`icatb_icaOptions.m`](GroupICAT/icatb/icatb_analysis_functions/icatb_icaOptions.m#L318-L416)).
It is dispatched through `icatb_runica_sbica`
([`icatb_icaAlgorithm.m`](GroupICAT/icatb/icatb_analysis_functions/icatb_icaAlgorithm.m#L168-L176)).

Spatially constrained ICA accepts one or more reference maps and exposes:

| Parameter | GIFT default | Role |
|---|---:|---|
| `loopnum` | `10` | learning passes |
| `threshold` | `0.08` | closeness constraint |
| `rho` | `1e-2` | positive contrast constant |
| `mu` | `10.8` | Lagrange multiplier |
| `gamma` | `0.02` | constraint penalty |
| `epsilon` | `1e-4` | stopping criterion |

These options are defined in
[`icatb_icaOptions.m`](GroupICAT/icatb/icatb_analysis_functions/icatb_icaOptions.m#L419-L476).
The estimator removes means, prewhitens both observed and reference signals, updates
constraint multipliers, and decorrelates the demixing matrix in
[`icatb_multi_fixed_ICA_R_Cor.m`](GroupICAT/icatb/icatb_analysis_functions/icatb_algorithms/icatb_multi_fixed_ICA_R_Cor.m#L1-L77).

Adaptive-reverse constrained ICA-EBM accepts reference columns and exposes
`opt_approach`, `W_init`, `WDiffStop=1e-6`, `alpha0=1`, and `maxIter=512`
([`icatb_icaOptions.m`](GroupICAT/icatb/icatb_analysis_functions/icatb_icaOptions.m#L851-L906)).
Its internal option structure additionally includes whitening, initialization, and
constraint-step settings
([`icatb_AR_Constrainguess_mated_ICA_EBM.m`](GroupICAT/icatb/icatb_analysis_functions/icatb_algorithms/icatb_AR_Constrainguess_mated_ICA_EBM.m#L67-L104)).

### Proposed API and safeguards

```python
run_constrained_ica(
    data,
    references,
    method="ica_r",
    reference_mask=None,
    normalize_references=True,
    match="correlation",
    **method_options,
) -> ConstrainedICAResult
```

The API must validate reference geometry, resampling history, mask agreement, component
count, and reference normalization. The result should contain achieved correlations and
constraint diagnostics, not only maps.

Sign orientation should be tied to the supplied reference: after estimation, a negatively
correlated source and its mixing column should both be multiplied by `-1`. This preserves
the reconstruction and makes reference similarity interpretable.

Python should provide NIfTI/template preparation. Rust should implement the array-level
constraint objective and updates after the Python version is validated.

### Validation requirements

- Synthetic references with known similarity and distractor sources.
- References that are absent, duplicated, nearly collinear, or sign-reversed.
- Reconstruction invariance after reference-based sign correction.
- MATLAB parity for final reference correlations and source assignment.
- Explicit tests that references cannot be silently resampled or remasked differently from
  the analyzed data.

## 7. Component normalization and ambiguity correction

ICA has unavoidable component permutation and per-component scale/sign ambiguity. The port
should treat normalization as a recorded post-estimation transform, never as an untracked
mutation.

### Sign and phase orientation

For real fMRI components, GIFT recenters each map, computes skewness, and flips both the
source row and mixing column when skewness is negative
([`icatb_calculateICA.m`](GroupICAT/icatb/icatb_analysis_functions/icatb_calculateICA.m#L801-L820)).
The port should expose this as `orient_by_skewness(S, A)` and verify that `A @ S` is
unchanged.

For complex components, GIFT applies a phase correction to the source, mixing matrix, and
demixing matrix together
([`icatb_calculateICA.m`](GroupICAT/icatb/icatb_analysis_functions/icatb_calculateICA.m#L603-L609)).
That behavior already exists in the Python and Rust complex ports and should remain
available through the shared normalization API.

Reference-guided methods should prefer reference-correlation orientation over skewness.
The precedence should be explicit:

1. reference orientation, when a reference is attached to the component;
2. complex phase orientation for complex sources;
3. skewness orientation for unconstrained real sources;
4. no orientation when requested by the user.

### Scaling schemes

`icatb_scaleICA` defines five schemes
([`icatb_scaleICA.m`](GroupICAT/icatb/icatb_analysis_functions/icatb_scaleICA.m#L1-L58)):

- no scaling;
- scale to original data units/percent signal change;
- z-scores;
- normalize maps by the average of their top 1% positive voxels and transfer that scale to
  time courses;
- scale maps by time-course standard deviation and time courses by maximum map intensity.

The detailed implementations are in
[`icatb_scaleICA.m`](GroupICAT/icatb/icatb_analysis_functions/icatb_scaleICA.m#L88-L209).
The global `SCALE_DEFAULT = 2` selects z-scores under GIFT's numeric indexing convention,
and `CENTER_IMAGES = 1` controls image-distribution centering
([`icatb_defaults.m`](GroupICAT/icatb/icatb_defaults.m#L487-L487),
[`icatb_defaults.m`](GroupICAT/icatb/icatb_defaults.m#L628-L634)). Ports should use named
enums rather than numeric offsets.

For a single run, map centering must be computed per component. The group calibration code
that derives offsets from mean maps across subjects should not be ported.

### Validation requirements

- Every sign/phase/scale transform must preserve `A @ S` unless the selected definition,
  such as z-scoring both outputs, intentionally changes units.
- Store the exact multiplicative factor and additive offset for every component.
- Define zero-variance and no-positive-voxel behavior explicitly.
- Test percent-signal-change scaling against data with known baselines.

## 8. Component quality control and labeling

This feature should produce machine-readable metrics first and optional classifiers second.
It should not depend on a GUI.

### Template labeling and matching

GIFT's component labeler accepts component images, a 4-D template, and a text label file,
then labels each component by its maximum template correlation
([`icatb_componentLabeller.m`](GroupICAT/icatb/icatb_helper_functions/icatb_componentLabeller.m#L1-L9),
[`icatb_componentLabeller.m`](GroupICAT/icatb/icatb_helper_functions/icatb_componentLabeller.m#L116-L169)).
The greedy-sort implementation instead builds the full component-template correlation
matrix and selects one-to-one matches
([`icatb_cls_greedy_sort_components.m`](GroupICAT/icatb/icatb_implementations/icatb_cls_greedy_sort_components.m#L1-L25),
[`icatb_cls_greedy_sort_components.m`](GroupICAT/icatb/icatb_implementations/icatb_cls_greedy_sort_components.m#L335-L378)).

The port should offer both:

- independent best-match labels, where multiple components may share a template;
- one-to-one assignment using the Hungarian algorithm, with greedy matching retained only
  as a MATLAB compatibility mode.

Outputs should include correlations, assignments, unmatched components/templates, mask
overlap, and resampling metadata.

### Noisecloud-style features

Noisecloud computes a standardized feature vector from spatial maps and time courses
([`noisecloud.m`](GroupICAT/icatb/toolbox/noisecloud/noisecloud.m#L1-L3),
[`noisecloud.m`](GroupICAT/icatb/toolbox/noisecloud/noisecloud.m#L135-L167)). Spatial
features include:

- kurtosis;
- atlas distribution;
- degree clustering;
- node distance;
- entropy;
- mirror symmetry;
- skewness;
- tissue overlap.

The calls are enumerated in
[`nc_spatial_features.m`](GroupICAT/icatb/toolbox/noisecloud/scripts/nc_spatial_features.m#L1-L31).
Temporal features include:

- autocorrelation;
- amplitude bins;
- dynamic range;
- entropy;
- high-frequency noise;
- kurtosis;
- mean;
- peak measures;
- power spectral density;
- skewness;
- energy ratio.

The calls are enumerated in
[`nc_temporal_features.m`](GroupICAT/icatb/toolbox/noisecloud/scripts/nc_temporal_features.m#L1-L35).
GIFT's classifier uses elastic-net logistic regression
([`noisecloud_classify.m`](GroupICAT/icatb/toolbox/noisecloud/noisecloud_classify.m#L1-L7)).

### Proposed implementation boundary

- Port metric extraction before porting a trained classifier.
- Use explicit atlas and tissue-mask inputs; do not hide downloads or registration inside
  numerical functions.
- Version every feature definition and preserve raw metrics before standardization.
- In Python, expose a table/data-frame result and optional scikit-learn classifier adapter.
- In Rust, implement inexpensive array metrics useful for batch acceleration; atlas-aware
  orchestration may remain in Python.

### Validation requirements

- Test every metric independently with synthetic maps/time courses.
- Compare feature vectors to MATLAB fixtures before training a classifier.
- Separate missing/undefined metrics from numeric zero.
- Report classifier version, training domain, TR assumptions, and decision probability.

## 9. Time-course preprocessing

Time-course preprocessing should be a reusable pipeline whose order is fixed and recorded.
GIFT's dFNC helper makes the intended order especially clear:

1. detrend;
2. regress nuisance covariates;
3. resample if TR harmonization is required;
4. despike;
5. filter.

The calls occur in that order in
[`icatb_compute_dfnc.m`](GroupICAT/icatb/icatb_helper_functions/icatb_compute_dfnc.m#L145-L200).
For a single run, resampling is optional and should be disabled unless the user explicitly
requests a new sampling grid.

### Detrending

`DETRENDNUMBER` selects four models
([`icatb_defaults.m`](GroupICAT/icatb/icatb_defaults.m#L398-L408)):

| Value | Regressors removed |
|---:|---|
| `0` | mean |
| `1` | mean and linear trend |
| `2` | one-cycle sine/cosine, linear trend, and mean |
| `3` | one- and two-cycle sine/cosine terms, linear trend, and mean |

The implementation builds and regresses these bases per data set in
[`icatb_detrend.m`](GroupICAT/icatb/icatb_analysis_functions/icatb_detrend.m#L1-L17)
and
[`icatb_detrend.m`](GroupICAT/icatb/icatb_analysis_functions/icatb_detrend.m#L70-L180).

### Nuisance regression

The API should accept an explicit design matrix, optional intercept behavior, and optional
scan selection. GIFT removes covariate variance in the helper beginning at
[`icatb_compute_dfnc.m`](GroupICAT/icatb/icatb_helper_functions/icatb_compute_dfnc.m#L445-L473).
The port must define behavior for collinear regressors and use a stable least-squares or
SVD solution with reported rank.

### Despiking

GIFT exposes three methods
([`icatb_defaults.m`](GroupICAT/icatb/icatb_defaults.m#L735-L746)):

- AFNI-style curve fitting, with `c = [2.5, 3]`;
- deviations from a smoothed reference time course, the current default;
- median filtering, with configurable order (`order = 4` in current defaults).

The dispatcher and method-specific implementations are in
[`icatb_despike_tc.m`](GroupICAT/icatb/icatb_helper_functions/icatb_despike_tc.m#L1-L48).
The port should start with the smoothed-reference method, then implement AFNI-style parity
and median filtering as named alternatives.

### Filtering

`icatb_filt_data` uses a fifth-order Butterworth filter followed by zero-phase `filtfilt`;
the cutoff is normalized by the Nyquist frequency
([`icatb_filt_data.m`](GroupICAT/icatb/icatb_helper_functions/icatb_filt_data.m#L1-L32)).
The current postprocessing default is `0.15 Hz`
([`icatb_defaults.m`](GroupICAT/icatb/icatb_defaults.m#L620-L623)). A scalar cutoff implies
low-pass filtering; a two-element cutoff can represent a band-pass under the underlying
Butterworth API. The port should validate all cutoffs against `(0, Nyquist)` and document
edge-padding behavior.

### Proposed API and validation

```python
preprocess_timecourses(
    timecourses,
    tr,
    detrend=1,
    confounds=None,
    despike="smooth",
    filter_band=None,
    resample_tr=None,
) -> PreprocessedTimecourses
```

The result should include the processed data, replaced-frame indices, design rank, filter
coefficients, resampling ratio, and an ordered provenance record.

Validation should use injected polynomial/sinusoidal drift, known nuisance signals,
synthetic spikes, and in-band/out-of-band sinusoids. Tests must verify operation order,
not merely the final shape.

## 10. Per-run component diagnostics

Diagnostics should consume one run's component maps and time courses and return structured
data. Plotting belongs in a separate Python layer.

### Spectral measures

`icatb_get_spectra` supports multitaper spectra or binned FFT spectra
([`icatb_get_spectra.m`](GroupICAT/icatb/icatb_helper_functions/icatb_get_spectra.m#L1-L70)).
Relevant defaults in `TIMECOURSE_POSTPROCESS` are
([`icatb_defaults.m`](GroupICAT/icatb/icatb_defaults.m#L602-L623)):

- method option `1` for multitaper;
- tapers `[3, 5]`;
- sampling frequency `1 / TR`;
- frequency band `[0, 1 / (2*TR)]`;
- configurable FFT length and number of bins;
- spectral summary limits `[0.1, 0.15]`;
- optional despiking and `0.15 Hz` filtering before FNC-style diagnostics.

`icatb_get_spec_stats` computes dynamic range and an fALFF-like ratio using configurable
frequency limits
([`icatb_get_spec_stats.m`](GroupICAT/icatb/icatb_mancovan_files/icatb_get_spec_stats.m#L1-L17)).
The port should preserve the MATLAB-compatible formula as a named compatibility metric,
while also offering conventional band definitions explicitly because the existing ratio's
`f < lower_limit` and `f > upper_limit` regions are easy to misinterpret.

### Distribution and signal metrics

Per component, compute at minimum:

- spatial and temporal mean, standard deviation, skewness, and kurtosis;
- temporal entropy and autocorrelation;
- peak and high-frequency-noise measures;
- positive/negative spatial extent under a stated threshold;
- map/time-course finite-value and zero-variance flags.

These overlap intentionally with the Noisecloud feature functions described in feature 8.
There should be one implementation of each numerical metric, reused by both diagnostics
and classifiers.

### Component sorting against models or templates

GIFT sorts temporal or spatial components by multiple regression, correlation, or kurtosis
([`icatb_sortComponents.m`](GroupICAT/icatb/icatb_analysis_functions/icatb_sortComponents.m#L1-L40),
[`icatb_sortComponents.m`](GroupICAT/icatb/icatb_analysis_functions/icatb_sortComponents.m#L109-L182)).
The single-run port should expose these as pure ranking functions returning score arrays,
sort indices, regression coefficients, partial correlations, and design metadata.

### Event averages and explained variance

`icatb_calculate_eventAvg` interpolates a component time course, extracts windows at
selected onsets, and averages them
([`icatb_calculate_eventAvg.m`](GroupICAT/icatb/icatb_helper_functions/icatb_calculate_eventAvg.m#L1-L43)).
The port should accept onsets in seconds or samples, validate windows at run boundaries,
and return both the mean and the number of contributing events at every lag.

`icatb_compute_percent_variance` regresses the data on a model and reports
`100 * (1 - SS_residual / SS_total)`
([`icatb_compute_percent_variance.m`](GroupICAT/icatb/icatb_helper_functions/icatb_compute_percent_variance.m#L1-L12),
[`icatb_compute_percent_variance.m`](GroupICAT/icatb/icatb_helper_functions/icatb_compute_percent_variance.m#L60-L73)).
For one run, diagnostics should report total variance explained by all retained components
and optionally component-wise partial contributions under a clearly stated decomposition.

### Output and validation

Python should return a tidy table with one row per component plus optional frequency and
event-average arrays. Rust should provide reusable scalar/vector kernels and a serializable
diagnostic result.

Validation should include analytic sinusoids, known event responses, heavy- and
light-tailed distributions, and exact low-rank reconstructions. Spectral frequency axes and
normalizations require direct MATLAB comparison.

## Recommended implementation sequence

The features have dependencies and should not be implemented as nine independent ports:

1. Define the shared single-run data/result contract.
2. Port preprocessing primitives and dense PCA/whitening.
3. Add real ICA-EBM and the end-to-end real pipeline.
4. Add sign/scale normalization and reconstruction-invariance tests.
5. Add FastICA, then Infomax.
6. Add IID MDL/AIC/KIC model-order estimation.
7. Add component diagnostics and template matching.
8. Add streamed/MPOWIT PCA for scale.
9. Add reference-guided ICA.
10. Add Noisecloud-compatible metrics and optional classification.
11. Add the remaining dimensionality and PCA methods only when validation fixtures and
    performance benchmarks justify them.

Across all stages, MATLAB should remain the behavioral oracle where the implementation is
well-defined. Python should be validated first, and Rust should then be compared with both
the Python implementation and the same immutable oracle fixtures. Component comparisons
must account for permutation and sign/scale ambiguity, while reconstruction, whitening,
diagnostic curves, and preprocessing outputs can generally be compared directly.
