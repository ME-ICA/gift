"""Export the oracle fixtures + per-unit parity fixtures as .npy for the Rust port.

Run from `python/`:
    micromamba run -n giftenv python tools/export_rust_fixtures.py

Python is the INTERMEDIATE ORACLE for Rust: for each unit we write a fixed input and
Python's own output, and the Rust test asserts agreement. Python is itself validated
against the MATLAB oracle, so agreement with Python transitively validates against MATLAB.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from complex_gift.nf_table import load_nf_table, simplified_ppval
from complex_gift.phase_correct import align_to_reference, correct_phase
from complex_gift.phase_mask import otsu_threshold, phase_quality_mask, quality_map

FIX = Path(__file__).resolve().parents[2] / "complex_ica_fixtures"
OUT = FIX / "npy"


def save(name, arr):
    np.save(OUT / f"{name}.npy", arr)
    print(f"  {name}.npy {getattr(arr, 'shape', ())} {getattr(arr, 'dtype', type(arr))}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1234)

    # --- ground truth + MATLAB oracle decompositions ---
    d = loadmat(FIX / "complex_sources.mat")
    save("cS", d["cS"]); save("A", d["A"]); save("cX", d["cX"])
    for est in ("cebm", "ncfastica"):
        o = loadmat(FIX / f"oracle_{est}.mat")
        save(f"oracle_{est}_W", o["W"])

    # --- nf_table: scalars + spline coefficients, per nonlinearity ---
    nf = load_nf_table(FIX / "nf_table.mat")
    for name, v in nf.items():
        save(f"{name}_scalars",
             np.array([v.min_EGx, v.max_EGx, v.critical_point, v.critical_point2]))
        for attr in ("pp", "pp_slope"):
            pp = getattr(v, attr)
            save(f"{name}_{attr}_breaks", pp.breaks)
            save(f"{name}_{attr}_coefs", pp.coefs)

    # --- parity: simplified_ppval (pure arithmetic -> elementwise comparable) ---
    xs_all, ys_all = [], []
    for name in [f"nf{i}" for i in range(1, 9)]:
        pp = nf[name].pp
        # sample inside the knot range AND outside it (exercises the clamp branches)
        xs = np.concatenate([
            np.linspace(pp.breaks[0] - 2.0, pp.breaks[-1] + 2.0, 61),
            np.linspace(pp.breaks[0], pp.breaks[-1], 40),
        ])
        xs_all.append(xs)
        ys_all.append(np.array([simplified_ppval(pp, float(x)) for x in xs]))
    save("parity_ppval_xs", np.stack(xs_all))     # (8, 101)
    save("parity_ppval_ys", np.stack(ys_all))     # (8, 101)

    # --- parity: phase mask (pure arithmetic -> elementwise comparable) ---
    V, T = 500, 80
    mag = 1.0 + 0.1 * rng.standard_normal((V, T))
    phi = np.empty((V, T))
    phi[:250] = 0.1 * rng.standard_normal((250, T))       # stable phase
    phi[250:] = 2 * np.pi * rng.random((250, T))          # random phase
    Z = mag * np.exp(1j * phi)
    mask, Q, tau = phase_quality_mask(Z)
    save("parity_mask_Z", Z)
    save("parity_mask_Q", Q)
    save("parity_mask_tau", np.array([tau]))
    save("parity_mask_mask", mask.astype(np.uint8))       # u8, not bool

    # --- parity: phase correction (pure arithmetic -> elementwise comparable) ---
    N, Vv, M = 4, 3000, 20
    S0 = rng.standard_normal((N, Vv)) + 1j * 0.05 * rng.standard_normal((N, Vv))
    A0 = rng.standard_normal((M, N)) + 1j * rng.standard_normal((M, N))
    theta_true = np.array([0.7, -1.1, 0.3, 2.0])
    S_in = S0 * np.exp(1j * theta_true)[:, None]
    A_in = A0 * np.exp(-1j * theta_true)[None, :]
    pc_mask = np.zeros(Vv, dtype=bool); pc_mask[: Vv // 2] = True
    S_out, A_out, theta_out = correct_phase(S_in.copy(), A_in.copy(), mask=pc_mask)
    save("parity_pc_S_in", S_in); save("parity_pc_A_in", A_in)
    save("parity_pc_mask", pc_mask.astype(np.uint8))
    save("parity_pc_S_out", S_out); save("parity_pc_A_out", A_out)
    save("parity_pc_theta", theta_out)

    # --- parity: group alignment ---
    S_ref = rng.standard_normal((3, 1500)) + 1j * 0.05 * rng.standard_normal((3, 1500))
    rot = np.array([0.9, -0.4, 2.2])
    S_subj = S_ref * np.exp(1j * rot)[:, None]
    S_al, th_al = align_to_reference(S_subj.copy(), S_ref)
    save("parity_align_S_ref", S_ref); save("parity_align_S_in", S_subj)
    save("parity_align_S_out", S_al); save("parity_align_theta", th_al)

    print(f"\nwrote fixtures to {OUT}")


if __name__ == "__main__":
    main()
