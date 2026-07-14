import nibabel as nib
import numpy as np

from complex_gift.complex_io import read_complex
from complex_gift.driver import run_from_files, unmask


def test_unmask_restores_full_volume_shape():
    """Masked (N, V_masked) maps expand back to full (N, *dims), zeros outside the mask."""
    dims = (4, 4, 2)
    V = int(np.prod(dims))
    mask = np.zeros(V, dtype=bool)
    mask[::2] = True                     # every other voxel
    S = np.arange(2 * mask.sum()).reshape(2, mask.sum()).astype(np.complex128) + 1j
    vols = unmask(S, mask, dims)
    assert vols.shape == (2, *dims)
    assert vols.dtype == np.complex128
    flat = vols.reshape(2, V)
    assert np.allclose(flat[:, mask], S)
    assert np.all(flat[:, ~mask] == 0)   # outside the mask is zero, not garbage


def _write_subject(tmp_path, name, rng, Smaps, phi, T, dims):
    """Write one subject as an R_/I_ NIfTI pair of shape (*dims, T)."""
    N, Vsig = Smaps.shape
    V = int(np.prod(dims))
    Vnoise = V - Vsig

    tc = rng.standard_normal((T, N))
    base = 100.0 + 10.0 * rng.random(Vsig)
    M = base[None, :] + 5.0 * (tc @ Smaps) + 2.0 * rng.standard_normal((T, Vsig))
    sig = M * np.exp(1j * phi)[None, :]                                   # (T, Vsig)
    noise = (0.5 + rng.random((T, Vnoise))) * np.exp(
        1j * 2 * np.pi * rng.random((T, Vnoise))
    )
    Z = np.concatenate([sig, noise], axis=1)                              # (T, V)

    vol = Z.T.reshape(*dims, T)                                           # (*dims, T)
    r = tmp_path / f"R_{name}.nii"
    i = tmp_path / f"I_{name}.nii"
    nib.save(nib.Nifti1Image(vol.real.astype(np.float64), np.eye(4)), str(r))
    nib.save(nib.Nifti1Image(vol.imag.astype(np.float64), np.eye(4)), str(i))
    return (r, i)


def test_run_from_files_end_to_end(tmp_path):
    """Files in -> complex group ICA -> component maps written back out as R_/I_ NIfTI."""
    rng = np.random.default_rng(41)
    dims = (10, 10, 4)          # 400 voxels
    V = int(np.prod(dims))
    N, Vsig, T, n_sub = 3, 300, 60, 3

    Smaps = rng.standard_normal((N, Vsig)) * np.abs(rng.standard_normal((N, Vsig))) ** 1.5
    Smaps /= np.abs(Smaps).max()
    phi = (rng.random(Vsig) * 20 - 10) / 180 * np.pi

    pairs = [
        _write_subject(tmp_path, f"sub{k:02d}", rng, Smaps, phi, T, dims)
        for k in range(n_sub)
    ]

    out_dir = tmp_path / "out"
    res, written = run_from_files(
        pairs, n_components=N, out_dir=out_dir,
        estimator="cebm", rng=np.random.default_rng(0),
    )

    # the mask kept the signal voxels and dropped the random-phase ones
    assert res.mask[:Vsig].mean() > 0.9
    assert res.mask[Vsig:].mean() < 0.05

    # component maps were written as R_/I_ pairs, at full volume shape, and read back
    assert len(written) == N
    for first, second in written:
        assert first.exists() and second.exists()
        z = read_complex(first, second)
        assert z.shape == dims
        assert z.dtype == np.complex128
        assert np.any(z != 0)                      # not an empty volume
        # voxels outside the mask are exactly zero in the written maps
        assert np.all(z.reshape(V)[~res.mask] == 0)
