# GIFT

MATLAB toolbox (GroupICAT/) plus a Python port under `python/` (the `gift` package, src-layout at `python/src/gift/`).

## Environment

The Python port uses the micromamba environment **`giftenv`**
(python 3.12, numpy, scipy, nibabel, scikit-image, pytest).

Run everything through it:

    micromamba run -n giftenv pytest python/tests -q

## Complex ICA

- Design spec: `docs/superpowers/specs/2026-07-13-complex-ica-port-design.md`
- Developer reference: `doc/complex_ica_porting_reference.md`
- MATLAB oracle artifacts: `complex_ica_fixtures/` (READ-ONLY - generated once from MATLAB)
