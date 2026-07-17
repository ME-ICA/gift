import logging

import nibabel as nb
import numpy as np
import pytest

from gift.complex_io import complex_file_pair, read_complex, write_complex


def _write(path, arr):
    nb.save(nb.Nifti1Image(arr.astype(np.float64), np.eye(4)), path)


def test_read_real_and_imaginary(tmp_path):
    rng = np.random.default_rng(8)
    re = rng.standard_normal((4, 4, 2))
    im = rng.standard_normal((4, 4, 2))
    _write(tmp_path / 'R_sub01.nii', re)
    _write(tmp_path / 'I_sub01.nii', im)
    z = read_complex(real_file=tmp_path / 'R_sub01.nii', imag_file=tmp_path / 'I_sub01.nii')
    assert z.dtype == np.complex128
    assert np.allclose(z.real, re)
    assert np.allclose(z.imag, im)


def test_read_magnitude_and_phase(tmp_path):
    rng = np.random.default_rng(9)
    mag = np.abs(rng.standard_normal((3, 3, 2))) + 0.5
    pha = rng.uniform(-np.pi, np.pi, (3, 3, 2))
    _write(tmp_path / 'Mag_sub01.nii', mag)
    _write(tmp_path / 'Phase_sub01.nii', pha)
    z = read_complex(mag_file=tmp_path / 'Mag_sub01.nii', phase_file=tmp_path / 'Phase_sub01.nii')
    assert np.allclose(np.abs(z), mag)
    assert np.allclose(np.angle(z), pha)


def test_read_requires_a_complete_pair():
    """Neither pair supplied is a caller error, not an empty result."""
    with pytest.raises(ValueError, match='neither pair'):
        read_complex()


@pytest.mark.parametrize(
    'kwargs',
    [
        {'mag_file': 'Mag_x.nii'},
        {'phase_file': 'Phase_x.nii'},
        {'real_file': 'R_x.nii'},
        {'imag_file': 'I_x.nii'},
    ],
)
def test_read_rejects_half_a_pair(kwargs):
    """Half a pair is unusable; fail loudly rather than guess the other half."""
    with pytest.raises(ValueError, match='must be given together'):
        read_complex(**kwargs)


def test_read_all_four_warns_and_prefers_real_imag(tmp_path, caplog):
    """All four supplied: real+imag wins, and the caller is told the other pair was dropped."""
    rng = np.random.default_rng(11)
    re = rng.standard_normal((3, 3, 2))
    im = rng.standard_normal((3, 3, 2))
    _write(tmp_path / 'R_sub01.nii', re)
    _write(tmp_path / 'I_sub01.nii', im)

    # A magnitude/phase pair describing a deliberately DIFFERENT complex array, so the
    # assertion below can only pass if real+imag actually won.
    _write(tmp_path / 'Mag_sub01.nii', np.full((3, 3, 2), 7.0))
    _write(tmp_path / 'Phase_sub01.nii', np.zeros((3, 3, 2)))

    with caplog.at_level(logging.WARNING, logger='gift.complex_io'):
        z = read_complex(
            mag_file=tmp_path / 'Mag_sub01.nii',
            phase_file=tmp_path / 'Phase_sub01.nii',
            real_file=tmp_path / 'R_sub01.nii',
            imag_file=tmp_path / 'I_sub01.nii',
        )

    assert np.allclose(z.real, re)
    assert np.allclose(z.imag, im)
    assert 'all four files supplied' in caplog.text


def test_write_read_roundtrip(tmp_path):
    rng = np.random.default_rng(10)
    z = rng.standard_normal((5, 5, 3)) + 1j * rng.standard_normal((5, 5, 3))
    f1, f2 = tmp_path / 'R_out.nii', tmp_path / 'I_out.nii'
    write_complex(z, f1, f2, affine=np.eye(4), complex_type='real&imaginary')
    z2 = read_complex(real_file=f1, imag_file=f2)
    assert np.allclose(z, z2)


def test_file_pair_naming():
    a, b = complex_file_pair('/data/sub01_run1.nii', naming=('R_', 'I_'))
    assert a.name == 'R_sub01_run1.nii'
    assert b.name == 'I_sub01_run1.nii'


def test_file_pair_requires_underscore():
    """GIFT's naming convention keys on an underscore; be explicit rather than silent."""
    with pytest.raises(ValueError, match='underscore'):
        complex_file_pair('/data/sub01.nii', naming=('R_', 'I_'))
