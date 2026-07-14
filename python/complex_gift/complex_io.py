"""Complex NIfTI I/O.

GIFT stores complex data as TWO ordinary NIfTI files per volume (image-domain
reconstructed data, not raw k-space), distinguished by a prefix around an underscore:
R_/I_ for real&imaginary, Mag_/Phase_ for magnitude&phase.
Mirrors icatb_loadData.m:79-85.
"""

from pathlib import Path

import nibabel as nib
import numpy as np

REAL_IMAG = "real&imaginary"
MAG_PHASE = "magnitude&phase"


def read_complex(first_path, second_path, complex_type=REAL_IMAG):
    """Assemble one complex array from GIFT's two-file representation."""
    a = np.asarray(nib.load(str(first_path)).get_fdata(), dtype=np.float64)
    b = np.asarray(nib.load(str(second_path)).get_fdata(), dtype=np.float64)
    if complex_type == REAL_IMAG:
        return (a + 1j * b).astype(np.complex128)
    if complex_type == MAG_PHASE:
        return (a * np.cos(b) + 1j * a * np.sin(b)).astype(np.complex128)
    raise ValueError(f"unknown complex_type {complex_type!r}")


def write_complex(data, first_path, second_path, affine, complex_type=REAL_IMAG):
    """Split a complex array back into two NIfTI files."""
    data = np.asarray(data, dtype=np.complex128)
    if complex_type == REAL_IMAG:
        a, b = data.real, data.imag
    elif complex_type == MAG_PHASE:
        a, b = np.abs(data), np.angle(data)
    else:
        raise ValueError(f"unknown complex_type {complex_type!r}")
    nib.save(nib.Nifti1Image(a, affine), str(first_path))
    nib.save(nib.Nifti1Image(b, affine), str(second_path))


def complex_file_pair(path, naming=("R_", "I_")):
    """Derive GIFT's two filenames from a base name (prefix around an underscore)."""
    p = Path(path)
    if "_" not in p.name:
        raise ValueError(
            f"complex file names must contain an underscore (GIFT convention): {p.name}"
        )
    return p.with_name(naming[0] + p.name), p.with_name(naming[1] + p.name)
