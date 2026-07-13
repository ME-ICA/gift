# Complex ICA in GIFT — MATLAB (Phase 0+1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Patch working complex-valued ICA (CEBM + nc-FastICA) into GIFT in place, and produce the shared lookup-table export and ground-truth fixtures that the later Python and Rust ports validate against.

**Architecture:** GIFT already has correct complex I/O, complex PCA/whitening, and de-meaning. This phase (1) vendors the two Adalí-lab estimators as `icatb_`-convention wrappers and registers them in the algorithm dispatcher, (2) generates deterministic complex fixtures + oracle decompositions + a portable `nf_table` export, then (3) un-hardcodes `dataType='complex'` and adds the two phase-specific steps (phase-quality mask, phase-ambiguity correction) as standalone, property-tested functions wired into GIFT's mask and component-finalization stages.

**Tech Stack:** MATLAB (R2019b+ assumed; user-run — no MATLAB/Octave in the dev environment). Vendored GPL-v3 MATLAB from `doc/complex_ica_reference/`. Tests are assert-based `.m` scripts run via `matlab -batch`.

## Global Constraints

- **License:** GPL v3. Every vendored/derived `.m` file keeps its GPL v3 header; new wrapper files carry a GPL v3 header and cite the source paper. (Spec §10.)
- **Estimator interface (both wrappers):** input `data` is `(N × T)` = components × volume (GIFT's convention); output is demixing matrix `W` `(N × N)` such that `icasig = W*data`, `A = pinv(W)`. (Spec §4.)
- **Fixtures are shared + version-controlled** at repo-root `complex_ica_fixtures/`; Python and Rust tracks read them unchanged. Never regenerate non-deterministically — always seed `rng`. (Spec §8.)
- **Whitening/estimator coupling:** `complex ica-ebm` ⇒ Hermitian whitening + `nf_table`; `complex nc-fastica` ⇒ its own internal SUT/pseudo-cov whitening, no table. (Spec §4.)
- **Test execution is user-run.** Every "Run the test" step is executed by the user in real MATLAB; the step states the exact command and expected output, and the implementer waits for the user to report PASS/FAIL before proceeding.
- **No MATLAB available to the agent** — do not attempt to run `.m` files locally; only write them.

**Directory conventions (created by this plan):**
- Wrappers, phase functions, vendored estimator code + `.mat` tables: `GroupICAT/icatb/icatb_analysis_functions/icatb_algorithms/complex_ica/`
- MATLAB test scripts: `GroupICAT/icatb/tests/complex_ica/`
- Shared fixtures + oracle outputs + table export: `complex_ica_fixtures/`

---

## Task 1: Vendor Complex ICA-EBM as a GIFT wrapper

**Files:**
- Create dir: `GroupICAT/icatb/icatb_analysis_functions/icatb_algorithms/complex_ica/`
- Copy (unchanged, keep GPL headers): all `.m` files from `doc/complex_ica_reference/complex_EBM/complex_ICA_EBM/` (i.e. `complex_ICA_EBM.m` defining `CEBM`, plus any helper subfunctions it defines) and `nf_table.mat` + `complex_nf_table.mat` into the new dir.
- Create: `.../complex_ica/icatb_complex_ica_ebm.m`
- Test: `GroupICAT/icatb/tests/complex_ica/test_complex_ica_ebm.m`

**Interfaces:**
- Consumes: `CEBM(X)` → `[W, min_cost, Ahat, Shat]` (vendored), `X` is `(N×T)`.
- Produces: `icatb_complex_ica_ebm(data)` → `W` `(N×N)` demixing matrix.

- [ ] **Step 1: Copy vendored files**

```bash
mkdir -p GroupICAT/icatb/icatb_analysis_functions/icatb_algorithms/complex_ica
cp doc/complex_ica_reference/complex_EBM/complex_ICA_EBM/*.m \
   doc/complex_ica_reference/complex_EBM/complex_ICA_EBM/nf_table.mat \
   doc/complex_ica_reference/complex_EBM/complex_ICA_EBM/complex_nf_table.mat \
   GroupICAT/icatb/icatb_analysis_functions/icatb_algorithms/complex_ica/
```

- [ ] **Step 2: Write the failing test**

`GroupICAT/icatb/tests/complex_ica/test_complex_ica_ebm.m`:

```matlab
% test_complex_ica_ebm — smoke + recovery test for the CEBM wrapper
function test_complex_ica_ebm()
addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'icatb_analysis_functions', 'icatb_algorithms', 'complex_ica')));
rng(42);
N = 4; T = 2000;
S = (randn(N,T) + 1i*randn(N,T));            % complex sources
A = randn(N,N) + 1i*randn(N,N);              % complex mixing
X = A*S;
W = icatb_complex_ica_ebm(X);
assert(isequal(size(W), [N N]), 'W must be N x N');
assert(~isreal(W), 'W must be complex');
% global matrix G = W*A should be a scaled permutation (one dominant entry per row)
G = abs(W*A);
G = G ./ max(G, [], 2);
dominant = sum(G > 0.5, 2);
assert(all(dominant == 1), 'W*A must be a scaled permutation (ISI check)');
disp('PASS test_complex_ica_ebm');
end
```

- [ ] **Step 3: Run the test to verify it FAILS** *(user-run)*

Run in MATLAB: `matlab -batch "cd('GroupICAT/icatb/tests/complex_ica'); test_complex_ica_ebm"`
Expected: FAIL — `Undefined function 'icatb_complex_ica_ebm'`.

- [ ] **Step 4: Write the wrapper**

`.../complex_ica/icatb_complex_ica_ebm.m`:

```matlab
function W = icatb_complex_ica_ebm(data)
%% Complex ICA by entropy bound minimization (wrapper around Adalí-lab CEBM).
% Input:  data - (components x volume) complex matrix (N x T)
% Output: W    - (N x N) complex demixing matrix; icasig = W*data
%
% Wraps CEBM (Li & Adalí 2010). Requires nf_table.mat on the path (shipped
% alongside this file). GPL v3 — see CEBM.m header; cite Li & Adalí 2010.
if isreal(data)
    error('icatb_complex_ica_ebm:realInput', 'Complex ICA-EBM requires complex-valued data.');
end
W = CEBM(data);
end
```

- [ ] **Step 5: Run the test to verify it PASSES** *(user-run)*

Run in MATLAB: `matlab -batch "cd('GroupICAT/icatb/tests/complex_ica'); test_complex_ica_ebm"`
Expected: prints `PASS test_complex_ica_ebm`, exit 0.

- [ ] **Step 6: Commit**

```bash
git add GroupICAT/icatb/icatb_analysis_functions/icatb_algorithms/complex_ica GroupICAT/icatb/tests/complex_ica/test_complex_ica_ebm.m
git commit -m "feat(complex-ica): vendor CEBM + icatb_complex_ica_ebm wrapper"
```

---

## Task 2: Vendor noncircular complex FastICA as a GIFT wrapper

**Files:**
- Copy (keep GPL header): `doc/complex_ica_reference/nonCircComplexFastICAsym.m` → `.../complex_ica/`
- Create: `.../complex_ica/icatb_complex_nc_fastica.m`
- Test: `GroupICAT/icatb/tests/complex_ica/test_complex_nc_fastica.m`

**Interfaces:**
- Consumes: `nonCircComplexFastICAsym(xold, typeStr)` → `[Ahat, shat]`, `xold` is `(n×m)`, `typeStr ∈ {'log','kurt','sqrt'}`.
- Produces: `icatb_complex_nc_fastica(data, typeStr)` → `W` `(N×N)`; `typeStr` optional, default `'log'`.

- [ ] **Step 1: Copy vendored file**

```bash
cp doc/complex_ica_reference/nonCircComplexFastICAsym.m \
   GroupICAT/icatb/icatb_analysis_functions/icatb_algorithms/complex_ica/
```

- [ ] **Step 2: Write the failing test**

`GroupICAT/icatb/tests/complex_ica/test_complex_nc_fastica.m`:

```matlab
function test_complex_nc_fastica()
addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'icatb_analysis_functions', 'icatb_algorithms', 'complex_ica')));
rng(7);
N = 4; T = 3000;
% noncircular complex sources: unequal real/imag variance
S = (randn(N,T) + 1i*0.3*randn(N,T));
A = randn(N,N) + 1i*randn(N,N);
X = A*S;
W = icatb_complex_nc_fastica(X, 'log');
assert(isequal(size(W), [N N]), 'W must be N x N');
G = abs(W*A); G = G ./ max(G, [], 2);
assert(all(sum(G > 0.5, 2) == 1), 'W*A must be a scaled permutation');
% default typeStr
W2 = icatb_complex_nc_fastica(X);
assert(isequal(size(W2), [N N]), 'default typeStr must work');
disp('PASS test_complex_nc_fastica');
end
```

- [ ] **Step 3: Run the test to verify it FAILS** *(user-run)*

Run in MATLAB: `matlab -batch "cd('GroupICAT/icatb/tests/complex_ica'); test_complex_nc_fastica"`
Expected: FAIL — `Undefined function 'icatb_complex_nc_fastica'`.

- [ ] **Step 4: Write the wrapper**

`.../complex_ica/icatb_complex_nc_fastica.m`:

```matlab
function W = icatb_complex_nc_fastica(data, typeStr)
%% Noncircular complex FastICA (wrapper around Adalí-lab nonCircComplexFastICAsym).
% Input:  data    - (components x volume) complex matrix (N x T)
%         typeStr - nonlinearity 'log' | 'kurt' | 'sqrt' (default 'log')
% Output: W       - (N x N) complex demixing matrix; icasig = W*data
%
% Wraps nonCircComplexFastICAsym (Novey & Adalí 2008). GPL v3 — see that
% file's header; cite Novey & Adalí 2008 (TSP).
if isreal(data)
    error('icatb_complex_nc_fastica:realInput', 'Noncircular complex FastICA requires complex-valued data.');
end
if (~exist('typeStr', 'var') || isempty(typeStr))
    typeStr = 'log';
end
[Ahat, ~] = nonCircComplexFastICAsym(data, typeStr);
W = pinv(Ahat);
end
```

- [ ] **Step 5: Run the test to verify it PASSES** *(user-run)*

Run in MATLAB: `matlab -batch "cd('GroupICAT/icatb/tests/complex_ica'); test_complex_nc_fastica"`
Expected: prints `PASS test_complex_nc_fastica`.

- [ ] **Step 6: Commit**

```bash
git add GroupICAT/icatb/icatb_analysis_functions/icatb_algorithms/complex_ica/nonCircComplexFastICAsym.m GroupICAT/icatb/icatb_analysis_functions/icatb_algorithms/complex_ica/icatb_complex_nc_fastica.m GroupICAT/icatb/tests/complex_ica/test_complex_nc_fastica.m
git commit -m "feat(complex-ica): vendor nc-FastICA + icatb_complex_nc_fastica wrapper"
```

---

## Task 3: Register both estimators in the algorithm dispatcher

**Files:**
- Modify: `GroupICAT/icatb/icatb_analysis_functions/icatb_icaAlgorithm.m` (name lists at :23, :30, :35; dispatch cases after :219)
- Test: `GroupICAT/icatb/tests/complex_ica/test_dispatch_complex.m`

**Interfaces:**
- Consumes: `icatb_complex_ica_ebm`, `icatb_complex_nc_fastica` (Tasks 1–2).
- Produces: dispatch names `'complex ica-ebm'` and `'complex nc-fastica'` in `icatb_icaAlgorithm`, returning `[icaAlgo]` list entries and, when called with data, `[W, A, icasig_tmp]` following the existing 3-output contract.

- [ ] **Step 1: Write the failing test**

`GroupICAT/icatb/tests/complex_ica/test_dispatch_complex.m`:

```matlab
function test_dispatch_complex()
addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'icatb_analysis_functions')));
% names appear in the algorithm list
algos = icatb_icaAlgorithm;
names = lower(cellstr(algos));
assert(any(strcmp(names, 'complex ica-ebm')), 'complex ica-ebm not registered');
assert(any(strcmp(names, 'complex nc-fastica')), 'complex nc-fastica not registered');
% dispatch runs and returns W, A, icasig with correct shapes
rng(1); N = 3; T = 1500;
S = randn(N,T) + 1i*randn(N,T); A0 = randn(N,N)+1i*randn(N,N); X = A0*S;
[W, A, icasig] = icatb_icaAlgorithm('complex ica-ebm', X);
assert(isequal(size(W), [N N]) && isequal(size(icasig), [N T]), 'ebm dispatch shapes');
[W2, A2, icasig2] = icatb_icaAlgorithm('complex nc-fastica', X);
assert(isequal(size(W2), [N N]) && isequal(size(icasig2), [N T]), 'ncfastica dispatch shapes');
disp('PASS test_dispatch_complex');
end
```

- [ ] **Step 2: Run the test to verify it FAILS** *(user-run)*

Run in MATLAB: `matlab -batch "cd('GroupICAT/icatb/tests/complex_ica'); test_dispatch_complex"`
Expected: FAIL — assertion `complex ica-ebm not registered`.

- [ ] **Step 3: Add names to the three `icaAlgo` lists**

In `icatb_icaAlgorithm.m`, append to the `char(...)` list at line 23 (fmri), 30 (smri), and 35 (eeg) the two new names. For the fmri list at :23–27, change the final argument so the list ends with:

```matlab
        'Adaptive Reverse Constrained IVA Gauss', 'Threshold Free Constrained IVA Gauss', ...
        'Complex ICA-EBM', 'Complex nc-FastICA');
```

Apply the equivalent append (`, 'Complex ICA-EBM', 'Complex nc-FastICA'` before the closing `)`) to the `:30` and `:35` lists.

- [ ] **Step 4: Add dispatch cases**

In the `switch(lower(selected_ica_algorithm))` block, immediately after the `case {'fbss', 'erbm'}` block that ends at line 219, insert:

```matlab
        case 'complex ica-ebm'
            %% Complex ICA by entropy bound minimization (Li & Adalí 2010)

            W = icatb_complex_ica_ebm(data);
            icasig_tmp = W*data;
            A = pinv(W);

        case 'complex nc-fastica'
            %% Noncircular complex FastICA (Novey & Adalí 2008)

            typeStr = 'log';
            if (~isempty(ICA_Options))
                tInd = strmatch('nonlinearity', lower(ICA_Options(1:2:end)), 'exact');
                if (~isempty(tInd)); typeStr = ICA_Options{2*tInd}; end
            end
            W = icatb_complex_nc_fastica(data, typeStr);
            icasig_tmp = W*data;
            A = pinv(W);
```

- [ ] **Step 5: Run the test to verify it PASSES** *(user-run)*

Run in MATLAB: `matlab -batch "cd('GroupICAT/icatb/tests/complex_ica'); test_dispatch_complex"`
Expected: prints `PASS test_dispatch_complex`.

- [ ] **Step 6: Commit**

```bash
git add GroupICAT/icatb/icatb_analysis_functions/icatb_icaAlgorithm.m GroupICAT/icatb/tests/complex_ica/test_dispatch_complex.m
git commit -m "feat(complex-ica): register Complex ICA-EBM and nc-FastICA in dispatcher"
```

---

## Task 4: Deterministic complex fixtures generator

**Files:**
- Create: `GroupICAT/icatb/tests/complex_ica/make_complex_fixtures.m`
- Output (git-tracked): `complex_ica_fixtures/complex_sources.mat` (`cS`, `A`, `cX`, meta)
- Test: `GroupICAT/icatb/tests/complex_ica/test_fixtures.m`

**Interfaces:**
- Consumes: nothing (self-contained; inspired by `doc/complex_ica_reference/simulate_complex_fmri_sources.m` but deterministic and plot-free).
- Produces: `complex_ica_fixtures/complex_sources.mat` with `cS` `(N×V)` complex sources, `A` `(N×N)` complex mixing, `cX = A*cS` `(N×V)` complex mixture, `N`, `V`, and `seed`.

- [ ] **Step 1: Write the generator**

`make_complex_fixtures.m`:

```matlab
function make_complex_fixtures()
%% Deterministic complex-source fixtures for cross-language validation.
% Writes complex_ica_fixtures/complex_sources.mat. Seeded — reproducible.
seed = 20260713; rng(seed);
N = 6; V = 4000;
% super-Gaussian complex sources with mildly noncircular structure
re = randn(N,V) .* (abs(randn(N,V)) .^ 1.5);
im = randn(N,V) .* (abs(randn(N,V)) .^ 1.5) .* 0.6;   % unequal variance -> noncircular
cS = re + 1i*im;
cS = cS - mean(cS, 2);                                 % zero-mean per source
A  = randn(N,N) + 1i*randn(N,N);
cX = A*cS;
outDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', '..', 'complex_ica_fixtures');
if (~exist(outDir, 'dir')); mkdir(outDir); end
save(fullfile(outDir, 'complex_sources.mat'), 'cS', 'A', 'cX', 'N', 'V', 'seed', '-v7');
fprintf('Wrote %s\n', fullfile(outDir, 'complex_sources.mat'));
end
```

*(Note: `-v7` MAT format is the most portable for later `scipy.io.loadmat` in the Python track. The `outDir` relative path resolves from `tests/complex_ica/` to repo root; verify the number of `..` segments matches your checkout depth and adjust if the printed path is wrong.)*

- [ ] **Step 2: Write the test**

`GroupICAT/icatb/tests/complex_ica/test_fixtures.m`:

```matlab
function test_fixtures()
here = fileparts(mfilename('fullpath'));
f = fullfile(here, '..', '..', '..', '..', 'complex_ica_fixtures', 'complex_sources.mat');
assert(exist(f, 'file') == 2, 'fixtures not generated — run make_complex_fixtures first');
d = load(f);
assert(~isreal(d.cS) && ~isreal(d.cX) && ~isreal(d.A), 'all arrays must be complex');
assert(isequal(size(d.cX), [d.N d.V]), 'cX shape');
assert(norm(d.cX - d.A*d.cS, 'fro') < 1e-9, 'cX must equal A*cS');
disp('PASS test_fixtures');
end
```

- [ ] **Step 3: Run — generate then test** *(user-run)*

Run in MATLAB:
`matlab -batch "cd('GroupICAT/icatb/tests/complex_ica'); make_complex_fixtures; test_fixtures"`
Expected: prints `Wrote .../complex_ica_fixtures/complex_sources.mat` then `PASS test_fixtures`.

- [ ] **Step 4: Commit (generator + fixture)**

```bash
git add GroupICAT/icatb/tests/complex_ica/make_complex_fixtures.m GroupICAT/icatb/tests/complex_ica/test_fixtures.m complex_ica_fixtures/complex_sources.mat
git commit -m "feat(complex-ica): deterministic complex fixtures generator + fixture"
```

---

## Task 5: Oracle export harness (decompositions + portable nf_table)

**Files:**
- Create: `GroupICAT/icatb/tests/complex_ica/export_oracle.m`
- Output (git-tracked): `complex_ica_fixtures/oracle_cebm.mat`, `complex_ica_fixtures/oracle_ncfastica.mat`, `complex_ica_fixtures/nf_table.csv`, `complex_ica_fixtures/complex_nf_table.csv`
- Test: `GroupICAT/icatb/tests/complex_ica/test_oracle_export.m`

**Interfaces:**
- Consumes: `complex_ica_fixtures/complex_sources.mat` (Task 4); `icatb_complex_ica_ebm`, `icatb_complex_nc_fastica` (Tasks 1–2); `nf_table.mat`.
- Produces: per-estimator `W`, `Ahat`, `Shat` saved as `.mat`; `nf1..nf8` lookup vectors exported to CSV. These are the ground truth the Python/Rust tracks diff against (Spec §8).

- [ ] **Step 1: Write the export harness**

`export_oracle.m`:

```matlab
function export_oracle()
here = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(here, '..', '..', 'icatb_analysis_functions', 'icatb_algorithms', 'complex_ica')));
fixDir = fullfile(here, '..', '..', '..', '..', 'complex_ica_fixtures');
d = load(fullfile(fixDir, 'complex_sources.mat'));

% --- oracle decompositions ---
W = icatb_complex_ica_ebm(d.cX); Ahat = pinv(W); Shat = W*d.cX;   %#ok<NASGU>
save(fullfile(fixDir, 'oracle_cebm.mat'), 'W', 'Ahat', 'Shat', '-v7');
W = icatb_complex_nc_fastica(d.cX, 'log'); Ahat = pinv(W); Shat = W*d.cX;
save(fullfile(fixDir, 'oracle_ncfastica.mat'), 'W', 'Ahat', 'Shat', '-v7');

% --- portable lookup tables ---
export_nf('nf_table.mat',         fullfile(fixDir, 'nf_table.csv'));
export_nf('complex_nf_table.mat', fullfile(fixDir, 'complex_nf_table.csv'));
disp('export_oracle done');
end

function export_nf(matName, csvPath)
S = load(matName);                 % nf_table.mat defines nf1..nf8 (and grid vars)
vars = fieldnames(S);
% write each variable as its own CSV column-block: name on header row, values below
fid = fopen(csvPath, 'w');
for k = 1:numel(vars)
    v = S.(vars{k});
    fprintf(fid, '%s\n', vars{k});
    fprintf(fid, '%.17g\n', v(:));
    fprintf(fid, '\n');
end
fclose(fid);
end
```

*(`nf_table.mat` is loaded by bare name — it is on the path via the addpath above. Inspect its actual variable names on first run; the harness exports whatever variables it contains, so no assumption about `nf1..nf8` is hard-coded in the writer.)*

- [ ] **Step 2: Write the test**

`test_oracle_export.m`:

```matlab
function test_oracle_export()
here = fileparts(mfilename('fullpath'));
fixDir = fullfile(here, '..', '..', '..', '..', 'complex_ica_fixtures');
for f = {'oracle_cebm.mat','oracle_ncfastica.mat','nf_table.csv','complex_nf_table.csv'}
    assert(exist(fullfile(fixDir, f{1}), 'file') == 2, ['missing ' f{1}]);
end
d = load(fullfile(fixDir, 'complex_sources.mat'));
o = load(fullfile(fixDir, 'oracle_cebm.mat'));
assert(isequal(size(o.W), [d.N d.N]), 'oracle W shape');
% W recovers sources: W*A is a scaled permutation
G = abs(o.W*d.A); G = G ./ max(G, [], 2);
assert(all(sum(G > 0.5, 2) == 1), 'oracle CEBM must recover sources');
disp('PASS test_oracle_export');
end
```

- [ ] **Step 3: Run — export then test** *(user-run)*

Run in MATLAB:
`matlab -batch "cd('GroupICAT/icatb/tests/complex_ica'); export_oracle; test_oracle_export"`
Expected: `export_oracle done` then `PASS test_oracle_export`.

- [ ] **Step 4: Commit**

```bash
git add GroupICAT/icatb/tests/complex_ica/export_oracle.m GroupICAT/icatb/tests/complex_ica/test_oracle_export.m complex_ica_fixtures/oracle_cebm.mat complex_ica_fixtures/oracle_ncfastica.mat complex_ica_fixtures/nf_table.csv complex_ica_fixtures/complex_nf_table.csv
git commit -m "feat(complex-ica): oracle decomposition + portable nf_table export"
```

> **Phase 0 exit criterion (Spec §9):** `complex_ica_fixtures/` populated and committed; estimators dispatch end-to-end in GIFT; `nf_table` export sanity-checked. Tasks 6–9 (Phase 1) can now proceed; the Python track can begin in parallel once these fixtures exist.

---

## Task 6: Un-hardcode `dataType` to allow `'complex'`

**Files:**
- Modify: `GroupICAT/icatb/icatb_setup_analysis.m:619` and `:1356`
- Modify: `GroupICAT/icatb/icatb_batch_files/icatb_read_batch_file.m:220`
- Test: `GroupICAT/icatb/tests/complex_ica/test_datatype_batch.m`

**Interfaces:**
- Consumes: batch input struct/file field `dataType` (+ optional `read_complex_images`, `write_complex_images`).
- Produces: `sesInfo.userInput.dataType` reflecting the input (`'real'` default, `'complex'` when requested) instead of a hardcoded `'real'`.

- [ ] **Step 1: Write the failing test**

`test_datatype_batch.m`:

```matlab
function test_datatype_batch()
addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', '..')));
% Minimal input file exercising the complex branch of the batch reader.
tmp = tempname; fid = fopen([tmp '.m'], 'w');
fprintf(fid, "dataType = 'complex';\n");
fprintf(fid, "read_complex_images = 'real&imaginary';\n");
fprintf(fid, "write_complex_images = 'real&imaginary';\n");
fclose(fid);
% Read just the dataType-relevant fields the way the batch reader does.
inp = icatb_eval_script([tmp '.m']);
assert(strcmpi(inp.dataType, 'complex'), 'batch input must carry dataType=complex');
disp('PASS test_datatype_batch');
end
```

*(This test targets the input-parsing contract: a batch input specifying `dataType='complex'` must be readable as complex. `icatb_eval_script` is GIFT's existing helper for evaluating an input `.m` into a struct; if the reader uses a different helper in your tree, match it. The point under test is that `dataType` is no longer forced to `'real'`.)*

- [ ] **Step 2: Run the test to verify it FAILS** *(user-run)*

Run in MATLAB: `matlab -batch "cd('GroupICAT/icatb/tests/complex_ica'); test_datatype_batch"`
Expected: FAIL (either the field is dropped or forced to real by the reader).

- [ ] **Step 3: Un-hardcode in the batch reader**

In `icatb_read_batch_file.m`, replace the hardcoded line 220:

```matlab
dataType = 'real'; read_complex_images = 'real&imaginary'; write_complex_images = 'real&imaginary';
```

with a read-from-input-with-default block:

```matlab
dataType = icatb_read_variable(inputData, 'dataType', 'char', 'real');
read_complex_images  = icatb_read_variable(inputData, 'read_complex_images',  'char', 'real&imaginary');
write_complex_images = icatb_read_variable(inputData, 'write_complex_images', 'char', 'real&imaginary');
```

*(Use the same accessor the surrounding code uses to pull fields from the parsed input — match the local variable name holding the parsed input struct, e.g. `inputData`/`sesInfo`. If no generic reader exists nearby, use a guarded `if isfield(...)` with the same three defaults.)*

- [ ] **Step 4: Un-hardcode in the GUI setup**

In `icatb_setup_analysis.m`, at `:619` and `:1356`, replace each hardcoded `dataType = 'real';` / `sesInfo.userInput.dataType = 'real';` with a read of any already-provided value, defaulting to `'real'`:

```matlab
if (~isfield(sesInfo.userInput, 'dataType') || isempty(sesInfo.userInput.dataType))
    sesInfo.userInput.dataType = 'real';
end
```

*(Preserve existing `read_complex_images`/`write_complex_images` assignments on the same lines; only stop forcing `dataType` to `'real'` when a value is already present.)*

- [ ] **Step 5: Run the test to verify it PASSES** *(user-run)*

Run in MATLAB: `matlab -batch "cd('GroupICAT/icatb/tests/complex_ica'); test_datatype_batch"`
Expected: prints `PASS test_datatype_batch`.

- [ ] **Step 6: Commit**

```bash
git add GroupICAT/icatb/icatb_setup_analysis.m GroupICAT/icatb/icatb_batch_files/icatb_read_batch_file.m GroupICAT/icatb/tests/complex_ica/test_datatype_batch.m
git commit -m "feat(complex-ica): allow dataType=complex (un-hardcode batch + setup)"
```

---

## Task 7: Phase-quality mask function + property test

**Files:**
- Create: `.../complex_ica/icatb_otsu_threshold.m`
- Create: `.../complex_ica/icatb_complex_phase_mask.m`
- Modify: `GroupICAT/icatb/icatb_helper_functions/icatb_createMask.m` (add complex Q-mask hook)
- Test: `GroupICAT/icatb/tests/complex_ica/test_phase_mask.m`

**Interfaces:**
- Consumes: complex data matrix `Z` `(V×T)`; optional magnitude mask `magMask` `(V×1 logical)`.
- Produces:
  - `icatb_otsu_threshold(x)` → scalar threshold `tau` (Otsu on a 256-bin histogram of vector `x`).
  - `icatb_complex_phase_mask(Z, magMask)` → `[mask, Q, tau]` where `Q(v) = |Σ_t Z(v,t)| / Σ_t |Z(v,t)|`, `tau = otsu(Q(magMask))`, `mask = magMask & (Q > tau)`.

- [ ] **Step 1: Write the property test**

`test_phase_mask.m`:

```matlab
function test_phase_mask()
addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'icatb_analysis_functions', 'icatb_algorithms', 'complex_ica')));
rng(3);
V = 500; T = 80;
% signal voxels: stable phase; noise voxels: random-walk phase
Z = zeros(V, T);
mag = 1 + 0.1*randn(V, T);
for v = 1:V
    if v <= 250
        phi = 0.1*randn(1, T);           % stable
    else
        phi = 2*pi*rand(1, T);           % random
    end
    Z(v, :) = mag(v, :) .* exp(1i*phi);
end
magMask = true(V, 1);

[mask, Q, tau] = icatb_complex_phase_mask(Z, magMask);
assert(mean(Q(1:250)) > mean(Q(251:end)) + 0.3, 'stable voxels must score higher Q');
assert(mean(mask(1:250)) > 0.8 && mean(mask(251:end)) < 0.2, 'mask must select stable voxels');

% PROPERTY: invariance to a constant per-voxel phase offset
offset = exp(1i * (2*pi*rand(V,1)));        % one constant phase per voxel
Z2 = Z .* repmat(offset, 1, T);
[mask2, Q2] = icatb_complex_phase_mask(Z2, magMask);
assert(max(abs(Q - Q2)) < 1e-10, 'Q must be invariant to constant per-voxel phase');
assert(isequal(mask, mask2), 'mask must be invariant to constant per-voxel phase');
disp('PASS test_phase_mask');
end
```

- [ ] **Step 2: Run the test to verify it FAILS** *(user-run)*

Run in MATLAB: `matlab -batch "cd('GroupICAT/icatb/tests/complex_ica'); test_phase_mask"`
Expected: FAIL — `Undefined function 'icatb_complex_phase_mask'`.

- [ ] **Step 3: Write the Otsu helper**

`.../complex_ica/icatb_otsu_threshold.m`:

```matlab
function tau = icatb_otsu_threshold(x)
%% Otsu threshold on a vector x (256-bin histogram). Returns scalar tau.
x = x(:); x = x(isfinite(x));
lo = min(x); hi = max(x);
if (hi <= lo); tau = lo; return; end
nbins = 256;
edges = linspace(lo, hi, nbins+1);
counts = histc(x, edges); counts = counts(1:nbins);
p = counts / sum(counts);
omega = cumsum(p);
mu = cumsum(p .* (1:nbins)');
muT = mu(end);
sigma_b = (muT*omega - mu).^2 ./ (omega .* (1 - omega) + eps);
[~, k] = max(sigma_b);
centers = (edges(1:nbins) + edges(2:nbins+1))' / 2;
tau = centers(k);
end
```

- [ ] **Step 4: Write the phase-mask function**

`.../complex_ica/icatb_complex_phase_mask.m`:

```matlab
function [mask, Q, tau] = icatb_complex_phase_mask(Z, magMask)
%% Phase-quality mask (Rodriguez 2011). Q(v) = |sum_t Z(v,t)| / sum_t |Z(v,t)|.
% Invariant to a constant per-voxel phase offset (measures temporal variation).
% Input:  Z       - (V x T) complex data
%         magMask - (V x 1 logical) magnitude/brain mask (optional; default all true)
% Output: mask (V x 1 logical), Q (V x 1), tau (scalar Otsu threshold)
[V, ~] = size(Z);
if (~exist('magMask', 'var') || isempty(magMask)); magMask = true(V, 1); end
magMask = logical(magMask(:));
num = abs(sum(Z, 2));
den = sum(abs(Z), 2) + eps;
Q = num ./ den;
tau = icatb_otsu_threshold(Q(magMask));
mask = magMask & (Q > tau);
end
```

- [ ] **Step 5: Run the test to verify it PASSES** *(user-run)*

Run in MATLAB: `matlab -batch "cd('GroupICAT/icatb/tests/complex_ica'); test_phase_mask"`
Expected: prints `PASS test_phase_mask`.

- [ ] **Step 6: Wire into `icatb_createMask.m`**

In `icatb_createMask.m`, the default-mask fmri branch (lines 82–125) assembles the magnitude-based `nonZeroInd`. After that branch computes `nonZeroInd` but before line 174 (`mask_ind = find(...)`), add a complex hook that intersects the phase-quality mask when data is complex. Insert after line 147 (`end` of the modality if/elseif/else), guarded:

```matlab
    %% Complex phase-quality mask (intersect with magnitude mask)
    if (strcmpi(dataType, 'complex'))
        % Reassemble full complex time series per file to score temporal phase stability.
        Zc = [];
        for i = 1:length(files)
            tempF = icatb_rename_4d_file(files(i).name);
            for nn = 1:size(tempF, 1)
                zc = icatb_loadData(tempF, dataType, complexInfo, 'read', nn);
                Zc = [Zc, zc(:)]; %#ok<AGROW>
            end
        end
        magMask = logical(nonZeroInd(:));
        pmask = icatb_complex_phase_mask(Zc, magMask);
        nonZeroInd = nonZeroInd & pmask;
    end
```

*(This reloads complex volumes to get the `(V×T)` matrix `Zc` the mask needs — `icatb_createMask`'s magnitude loop only keeps first-timepoint magnitudes. If memory is a concern for large data, this hook is the documented place to stream it; for Phase 1 correctness the straightforward assembly above is acceptable. Verify `complexInfo` is in scope here — it is a function argument.)*

- [ ] **Step 7: Commit**

```bash
git add GroupICAT/icatb/icatb_analysis_functions/icatb_algorithms/complex_ica/icatb_otsu_threshold.m GroupICAT/icatb/icatb_analysis_functions/icatb_algorithms/complex_ica/icatb_complex_phase_mask.m GroupICAT/icatb/icatb_helper_functions/icatb_createMask.m GroupICAT/icatb/tests/complex_ica/test_phase_mask.m
git commit -m "feat(complex-ica): phase-quality mask (Otsu) + createMask hook"
```

---

## Task 8: Phase-ambiguity correction function + property tests

**Files:**
- Create: `.../complex_ica/icatb_complex_phase_correct.m`
- Modify: `GroupICAT/icatb/icatb_analysis_functions/icatb_calculateICA.m` (apply correction near the complex-write block, ~:623–655)
- Test: `GroupICAT/icatb/tests/complex_ica/test_phase_correct.m`

**Interfaces:**
- Consumes: sources `S` `(N×V)` complex, mixing `A` `(M×N)` complex, optional `mask` `(V×1 logical)`.
- Produces: `icatb_complex_phase_correct(S, A, mask)` → `[S, A, theta]` where each component `k` is rotated `S(k,:) *= exp(1i*theta_k)`, `A(:,k) *= exp(-1i*theta_k)` with `theta_k = -0.5*angle(sum(S(k,mask).^2))`, then a residual π sign fix by skewness of the real part. `A*S` is preserved.

- [ ] **Step 1: Write the property tests**

`test_phase_correct.m`:

```matlab
function test_phase_correct()
addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'icatb_analysis_functions', 'icatb_algorithms', 'complex_ica')));
rng(5);
N = 4; V = 3000; M = 20;
% real-axis-aligned sources (elongated along real axis), then inject a known rotation
Sbase = (randn(N,V)) + 1i*(0.05*randn(N,V));      % energy mostly real
A0 = randn(M,N) + 1i*randn(M,N);
theta_true = [0.7; -1.1; 0.3; 2.0];
Srot = Sbase .* exp(1i*repmat(theta_true, 1, V));
Arot = A0 .* exp(-1i*repmat(theta_true.', M, 1)); % keep A*S invariant
mask = true(V,1);

% PROPERTY 1: X = A*S preserved by the whole correction
X_before = Arot*Srot;
[Sc, Ac, theta] = icatb_complex_phase_correct(Srot, Arot, mask);
assert(norm(X_before - Ac*Sc, 'fro') / norm(X_before, 'fro') < 1e-10, 'A*S must be preserved');

% PROPERTY 2: corrected sources are real-axis aligned (imag energy minimized)
imag_frac = sum(imag(Sc).^2, 2) ./ sum(abs(Sc).^2, 2);
assert(all(imag_frac < 0.05), 'corrected sources must concentrate energy on the real axis');

% PROPERTY 3: recovered rotation cancels the injected one (up to pi sign)
resid = mod(angle(sum(Sc.^2, 2)), pi);
assert(all(min(resid, pi - resid) < 1e-3), 'residual major-axis angle ~ 0 mod pi');
disp('PASS test_phase_correct');
end
```

- [ ] **Step 2: Run the test to verify it FAILS** *(user-run)*

Run in MATLAB: `matlab -batch "cd('GroupICAT/icatb/tests/complex_ica'); test_phase_correct"`
Expected: FAIL — `Undefined function 'icatb_complex_phase_correct'`.

- [ ] **Step 3: Write the correction function**

`.../complex_ica/icatb_complex_phase_correct.m`:

```matlab
function [S, A, theta] = icatb_complex_phase_correct(S, A, mask)
%% Phase-ambiguity correction (Rodriguez 2012). Rotate each component so energy
% is maximally concentrated on the real axis; apply inverse to A so A*S is preserved.
% Input:  S    - (N x V) complex sources
%         A    - (M x N) complex mixing
%         mask - (V x 1 logical) high-quality voxels for the estimate (optional)
% Output: S, A (rotated), theta (N x 1 applied rotations, incl. pi sign fix)
[N, V] = size(S);
if (~exist('mask', 'var') || isempty(mask)); mask = true(V, 1); end
mask = logical(mask(:));
theta = zeros(N, 1);
for k = 1:N
    sk = S(k, :);
    smk = sk(mask);
    th = -0.5 * angle(sum(smk.^2));      % orient major axis to real axis
    sk = sk * exp(1i*th);
    % residual pi (sign) ambiguity: make real-part skewness positive
    re = real(sk(mask));
    sg = mean((re - mean(re)).^3);
    if (sg < 0); th = th + pi; sk = sk * exp(1i*pi); end
    S(k, :) = sk;
    A(:, k) = A(:, k) * exp(-1i*th);
    theta(k) = th;
end
end
```

- [ ] **Step 4: Run the test to verify it PASSES** *(user-run)*

Run in MATLAB: `matlab -batch "cd('GroupICAT/icatb/tests/complex_ica'); test_phase_correct"`
Expected: prints `PASS test_phase_correct`.

- [ ] **Step 5: Wire into `icatb_calculateICA.m`**

In `icatb_calculateICA.m`, the complex-write path is guarded by `WRITE_COMPLEX_IMAGES` (assigned at :623). Locate where the final `icasig` and `A` are available for complex data just before they are split/written (~:637–655). Immediately before that write, apply the correction:

```matlab
    if (strcmpi(sesInfo.userInput.dataType, 'complex'))
        % Fix per-component phase ambiguity on high-quality (masked) voxels.
        pcMask = true(size(icasig, 2), 1);
        if (isfield(sesInfo, 'mask_ind') && ~isempty(sesInfo.mask_ind))
            pcMask = true(size(icasig, 2), 1);   % icasig already restricted to mask_ind
        end
        [icasig, A] = icatb_complex_phase_correct(icasig, A, pcMask);
    end
```

*(`icasig` here is `(components × voxels)` = `(N×V)` and `A` is the mixing whose product reconstructs the data — matching the function's `(N×V)` / `(M×N)` contract. `icasig` columns already correspond to in-mask voxels, so the default all-true `pcMask` is correct; refine only if you carry a within-mask quality sub-selection. Confirm the variable names `icasig`/`A` at this point in your tree and adjust if the local names differ.)*

- [ ] **Step 6: Commit**

```bash
git add GroupICAT/icatb/icatb_analysis_functions/icatb_algorithms/complex_ica/icatb_complex_phase_correct.m GroupICAT/icatb/icatb_analysis_functions/icatb_calculateICA.m GroupICAT/icatb/tests/complex_ica/test_phase_correct.m
git commit -m "feat(complex-ica): phase-ambiguity correction + calculateICA hook"
```

---

## Task 9: End-to-end complex analysis smoke test

**Files:**
- Create: `GroupICAT/icatb/tests/complex_ica/test_end_to_end_complex.m`
- Create (test asset): `GroupICAT/icatb/tests/complex_ica/write_complex_test_nifti.m`

**Interfaces:**
- Consumes: the full patched pipeline (Tasks 1–8): complex I/O, `dataType='complex'`, phase mask, complex ICA dispatch, phase correction.
- Produces: proof that a batch complex ICA run completes and writes split complex outputs.

- [ ] **Step 1: Write a helper that writes R_/I_ NIfTI test volumes**

`write_complex_test_nifti.m`:

```matlab
function outDir = write_complex_test_nifti(outDir)
%% Write a tiny synthetic complex fMRI dataset as R_/I_ NIfTI pairs for one subject.
if (~exist('outDir', 'var') || isempty(outDir)); outDir = tempname; end
if (~exist(outDir, 'dir')); mkdir(outDir); end
dim = [8 8 4]; T = 30; rng(11);
V = prod(dim);
cS = (randn(3,V) + 1i*0.1*randn(3,V));
tc = randn(T,3) + 1i*0.1*randn(T,3);
cX = tc*cS;                                 % (T x V) complex
for t = 1:T
    vol = reshape(cX(t, :), dim);
    icatb_write_nifti_data(fullfile(outDir, sprintf('R_sub01_%03d.nii', t)), makeHdr(dim), real(vol));
    icatb_write_nifti_data(fullfile(outDir, sprintf('I_sub01_%03d.nii', t)), makeHdr(dim), imag(vol));
end
end

function H = makeHdr(dim)
H = struct('dim', dim, 'dt', [16 0], 'mat', eye(4));  % adapt to icatb_write_nifti_data's expected header
end
```

*(Header construction must match whatever `icatb_write_nifti_data` — or the NIfTI writer available in your GIFT tree — expects. If that helper's signature differs, use the writer GIFT uses elsewhere for test data; the essential requirement is R_/I_ prefixed pairs per §2 of the reference doc.)*

- [ ] **Step 2: Write the end-to-end test**

`test_end_to_end_complex.m`:

```matlab
function test_end_to_end_complex()
here = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(here, '..', '..')));
dataDir = write_complex_test_nifti();
outDir = tempname; mkdir(outDir);

% Build a minimal complex batch input struct and run setup + analysis.
sesInfo = struct();
sesInfo.userInput.pwd = outDir;
sesInfo.userInput.prefix = 'cxtest';
sesInfo.userInput.dataType = 'complex';
sesInfo.userInput.read_complex_images  = 'real&imaginary';
sesInfo.userInput.write_complex_images = 'real&imaginary';
sesInfo.userInput.algorithm = 'complex ica-ebm';
sesInfo.userInput.numComp = 3;
sesInfo.userInput.files = struct('name', spm_select_or_dir(dataDir, 'R_sub01'));

% Run the reduction + ICA + back-recon path GIFT uses in batch.
sesInfo = icatb_runAnalysis(sesInfo, 1);   % adapt to the batch entry point in your tree

% Assert split complex outputs were written (R_/I_ component maps).
comp = dir(fullfile(outDir, '*_component_ica_*.nii'));
assert(~isempty(comp), 'no complex component maps written');
disp('PASS test_end_to_end_complex');
end
```

*(This task is the integration proof; its exact wiring depends on GIFT's batch entry point (`icatb_runAnalysis` / `icatb_batch_file_run`) and file-selection helpers in your tree. Treat the specific calls as adapt-to-your-version — the assertion that matters is: a `dataType='complex'` run with `algorithm='complex ica-ebm'` completes and writes split complex component maps. If full batch wiring is heavier than warranted here, reduce to driving `icatb_dataReduction` → `icatb_calculateICA` directly on the loaded complex matrix.)*

- [ ] **Step 3: Run the test** *(user-run)*

Run in MATLAB: `matlab -batch "cd('GroupICAT/icatb/tests/complex_ica'); test_end_to_end_complex"`
Expected: prints `PASS test_end_to_end_complex` (a real complex group analysis ran inside GIFT).

- [ ] **Step 4: Commit**

```bash
git add GroupICAT/icatb/tests/complex_ica/test_end_to_end_complex.m GroupICAT/icatb/tests/complex_ica/write_complex_test_nifti.m
git commit -m "test(complex-ica): end-to-end complex analysis smoke test"
```

> **Phase 1 exit criterion (Spec §9):** a real complex group analysis runs inside GIFT from batch; phase-step property tests (Tasks 7–8) pass. Deliverable: patched GIFT with complex ICA + committed fixtures/tables. Next: Phase 2 (Python), which consumes `complex_ica_fixtures/` as its oracle.

---

## Notes for the implementer

- **You cannot run MATLAB in the dev environment.** Write every `.m` exactly as specified; the user runs each "Run the test" step and reports PASS/FAIL. Do not mark a step done until the user confirms.
- **Adapt-to-your-tree markers** (Tasks 6–9) call out GIFT internals whose exact names/signatures may drift by release (`icatb_read_variable`, `icatb_eval_script`, `icatb_write_nifti_data`, the batch entry point, `icasig`/`A` locals). Verify each against the actual file before editing; the surrounding code shows the local conventions.
- **GPL v3 headers** stay on every vendored and derived file (Global Constraints).
- **Fixtures are the contract** for Phases 2–3 — never regenerate them non-deterministically or hand-edit the `.mat`/CSV; if a fixture must change, re-run its generator (seeded) and re-commit both generator and output together.
