# Adalı Lab (MLSP, UMBC) Complex ICA — Reference Implementations

Original MATLAB implementations from the **MLSP Lab, University of Maryland,
Baltimore County (UMBC)**, retrieved for use as **reference / correctness oracles**
when porting complex-valued ICA to other languages. See
[`../complex_ica_porting_reference.md`](../complex_ica_porting_reference.md) for how they
fit the port.

- **Source:** <https://mlsp.umbc.edu/resources.html> (files under `https://mlsp.umbc.edu/codes/`)
- **License:** GNU General Public License v3.0 (embedded in each file's header).
  Porting these algorithms produces a derivative work — this port is therefore
  **GPL v3**.
- **Attribution:** the lab asks that you cite the papers listed below in any related work.

> These files are **not** wired into GIFT and are not called by any GIFT code. They live
> here purely as a reference for the port.

## Contents

| Path | Algorithm | Entry function | Paper |
|------|-----------|----------------|-------|
| `complex_EBM/complex_ICA_EBM/complex_ICA_EBM.m` | Complex ICA-EBM (entropy bound minimization) | `CEBM(X)` | Li & Adalı 2010 |
| `PackageCERBM/PackageCodeCERBM/CERBM.m` | Complex ICA-ERBM (entropy *rate* bound min.; non-Gaussianity + nonwhiteness + noncircularity) | `CERBM(X, Lite, p)` | Fu, Phlypo, Anderson & Adalı 2015 |
| `PackageCERBM/PackageCodeCERBM/CEBM.m` | Complex ICA-EBM (bundled copy used by CERBM demo) | `CEBM(X)` | Li & Adalı 2010 |
| `nonCircComplexFastICAsym.m` | Noncircular complex FastICA (symmetric) | `nonCircComplexFastICAsym(X, typeStr)` | Novey & Adalı 2008 (TSP) |
| `TCMNsym.m` | T-CMN — complex ICA by negentropy maximization | `doCMNsym(X, typeStr, inVal)` | Novey & Adalı 2008 (NN) |
| `ACMNsym.m` | Adaptable complex maximization of non-Gaussianity | `ACMNsym(X, typeStr)` | Adalı lab |
| `CQAMsym.m` | Complex ICA for QAM constellations | `CQAMsym(X, numStars, sigSq)` | Adalı lab |
| `simulate_complex_fmri_sources.m` | Complex-valued fMRI-like source/data generator (script) | — | Adalı lab |
| `*/demo*.m`, `*/nf_table.mat`, `*/complex_nf_table.mat` | Demos and precomputed lookup tables | — | — |

## Interface conventions

- **Data orientation:** input `X` is `(N × T)` — `N` = number of sources/channels
  (rows), `T` = samples (columns). For fMRI spatial ICA this is the PCA-reduced data
  `(numComp × voxels)`, matching GIFT's convention (ICA runs on the reduced matrix).
- **Outputs:** demixing matrix `W`, and typically `Ahat` (mixing) and `Shat = W*X`
  (sources).
- **`nonCircComplexFastICAsym` / `TCMNsym`** take a nonlinearity selector string
  (`'log'`, `'kurt'`, `'sqrt'` for nc-FastICA; `cosh`/`pow`/… for T-CMN).

## ⚠️ Porting note: the `.mat` lookup tables are a dependency

`CEBM` / `CERBM` load **precomputed lookup tables** (`nf_table.mat`,
`complex_nf_table.mat`) that parameterize the entropy-bound nonlinearities. A faithful
port must either (a) ship the same tables (export them to a portable format — `.npy`,
CSV, etc.) or (b) regenerate them from the measure definitions in the papers. Don't
overlook these — the algorithm is not self-contained without them.

## References

- Li, X.-L. & Adalı, T. (2010). *Complex independent component analysis by entropy bound
  minimization.* IEEE Trans. Circuits Syst. I, 57(7):1417–1430.
- Fu, G.-S., Phlypo, R., Anderson, M. & Adalı, T. (2015). *Complex independent component
  analysis using three types of diversity: non-Gaussianity, nonwhiteness, and
  noncircularity.* IEEE Trans. Signal Process. 63(3):794–805.
- Novey, M. & Adalı, T. (2008). *On extending the complex FastICA algorithm to
  noncircular sources.* IEEE Trans. Signal Process. 56(5):2148–2154.
- Novey, M. & Adalı, T. (2008). *Complex ICA by negentropy maximization.* IEEE Trans.
  Neural Netw. 19(4):596–609.
