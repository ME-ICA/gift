import nibabel as nib
import numpy as np
import pytest

from complex_gift.complex_io import complex_file_pair, read_complex, write_complex


def _write(path, arr):
    nib.save(nib.Nifti1Image(arr.astype(np.float64), np.eye(4)), path)


def test_read_real_and_imaginary(tmp_path):
    rng = np.random.default_rng(8)
    re = rng.standard_normal((4, 4, 2))
    im = rng.standard_normal((4, 4, 2))
    _write(tmp_path / "R_sub01.nii", re)
    _write(tmp_path / "I_sub01.nii", im)
    z = read_complex(tmp_path / "R_sub01.nii", tmp_path / "I_sub01.nii",
                     complex_type="real&imaginary")
    assert z.dtype == np.complex128
    assert np.allclose(z.real, re) and np.allclose(z.imag, im)


def test_read_magnitude_and_phase(tmp_path):
    rng = np.random.default_rng(9)
    mag = np.abs(rng.standard_normal((3, 3, 2))) + 0.5
    pha = rng.uniform(-np.pi, np.pi, (3, 3, 2))
    _write(tmp_path / "Mag_sub01.nii", mag)
    _write(tmp_path / "Phase_sub01.nii", pha)
    z = read_complex(tmp_path / "Mag_sub01.nii", tmp_path / "Phase_sub01.nii",
                     complex_type="magnitude&phase")
    assert np.allclose(np.abs(z), mag)
    assert np.allclose(np.angle(z), pha)


def test_write_read_roundtrip(tmp_path):
    rng = np.random.default_rng(10)
    z = rng.standard_normal((5, 5, 3)) + 1j * rng.standard_normal((5, 5, 3))
    f1, f2 = tmp_path / "R_out.nii", tmp_path / "I_out.nii"
    write_complex(z, f1, f2, affine=np.eye(4), complex_type="real&imaginary")
    z2 = read_complex(f1, f2, complex_type="real&imaginary")
    assert np.allclose(z, z2)


def test_file_pair_naming():
    a, b = complex_file_pair("/data/sub01_run1.nii", naming=("R_", "I_"))
    assert a.name == "R_sub01_run1.nii"
    assert b.name == "I_sub01_run1.nii"


def test_file_pair_requires_underscore():
    """GIFT's naming convention keys on an underscore; be explicit rather than silent."""
    with pytest.raises(ValueError, match="underscore"):
        complex_file_pair("/data/sub01.nii", naming=("R_", "I_"))
