"""Complex NIfTI I/O."""

import logging
from pathlib import Path

import nibabel as nb
import numpy as np

LOGGER = logging.getLogger(__name__)

REAL_IMAG = 'real&imaginary'
MAG_PHASE = 'magnitude&phase'


def read_complex(mag_file=None, phase_file=None, real_file=None, imag_file=None):
    """Assemble one complex array from GIFT's two-file representation.

    Exactly one representation must be supplied: magnitude and phase, or real and
    imaginary. The representation is inferred from which arguments are given.

    Parameters
    ----------
    mag_file, phase_file : str or pathlib.Path or None, optional
        Paths to the magnitude and phase NIfTI images. Must be given together.
        Phase is assumed to be in radians.
    real_file, imag_file : str or pathlib.Path or None, optional
        Paths to the real and imaginary NIfTI images. Must be given together.

    Returns
    -------
    numpy.ndarray of complex128
        The assembled complex array, with the same shape as the input images.

    Raises
    ------
    ValueError
        If neither complete pair is supplied, or if a pair is only half supplied
        (e.g. ``mag_file`` without ``phase_file``).

    Notes
    -----
    GIFT stores complex data as two ordinary NIfTI files per volume, holding
    image-domain reconstructed data rather than raw k-space. This function takes the two
    paths explicitly, so it is agnostic to how they are named; the naming conventions
    below matter only to :func:`complex_file_pair` and to the file-level driver.

    GIFT distinguishes the two files by an underscore-delimited tag, and its defaults use
    a *different* tag for reading than for writing (``icatb_defaults.m``,
    ``READ_NAMING_COMPLEX_IMAGES`` vs. ``WRITE_NAMING_COMPLEX_IMAGES``). Reading uses a
    suffix -- ``_R``/``_I`` for real and imaginary, ``_CM``/``_CP`` for magnitude and
    phase -- while writing uses a prefix -- ``R_``/``I_`` and ``Mag_``/``Phase_``. On
    read, GIFT accepts either prefix or suffix; on write it emits the prefix form only
    (see ``icatb_loadData.m`` lines 13-15).

    The complex value itself is assembled exactly as ``icatb_loadData.m`` lines 79-87:
    ``complex(real, imag)`` for the real/imaginary pair, and
    ``complex(mag.*cos(phase), mag.*sin(phase))`` for magnitude and phase. The latter is
    :math:`z = m \\cos\\varphi + i\\,m \\sin\\varphi`, equivalently :math:`z = m e^{i\\varphi}`.

    If all four files are supplied, a warning is logged and the real and imaginary pair
    is used. That pair needs no trigonometry, so it reproduces the stored values
    exactly, whereas magnitude and phase reconstruct them.
    """
    has_mag_phase = mag_file is not None and phase_file is not None
    has_real_imag = real_file is not None and imag_file is not None

    if (mag_file is None) != (phase_file is None):
        raise ValueError(
            'read_complex: mag_file and phase_file must be given together; got '
            f'mag_file={mag_file!r}, phase_file={phase_file!r}.'
        )
    if (real_file is None) != (imag_file is None):
        raise ValueError(
            'read_complex: real_file and imag_file must be given together; got '
            f'real_file={real_file!r}, imag_file={imag_file!r}.'
        )
    if not has_mag_phase and not has_real_imag:
        raise ValueError(
            'read_complex: supply either mag_file and phase_file, or real_file and '
            'imag_file; got neither pair.'
        )
    if has_mag_phase and has_real_imag:
        LOGGER.warning(
            'read_complex: all four files supplied; using real_file and imag_file, '
            'ignoring mag_file and phase_file.'
        )

    if has_real_imag:
        real = _load(real_file)
        imag = _load(imag_file)
        return (real + 1j * imag).astype(np.complex128)

    mag = _load(mag_file)
    phase = _load(phase_file)
    return (mag * np.cos(phase) + 1j * mag * np.sin(phase)).astype(np.complex128)


def _load(path):
    """Load one NIfTI image as a float64 array.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the NIfTI image.

    Returns
    -------
    numpy.ndarray of float64
        The image data.
    """
    return np.asarray(nb.load(str(path)).get_fdata(), dtype=np.float64)


def write_complex(data, first_path, second_path, affine, complex_type=REAL_IMAG):
    """Split a complex array into GIFT's two-file representation and write it out.

    Parameters
    ----------
    data : array_like of complex
        The complex array to write.
    first_path, second_path : str or pathlib.Path
        Output paths. These receive the real and imaginary parts when ``complex_type``
        is ``'real&imaginary'``, or the magnitude and phase when it is
        ``'magnitude&phase'``.
    affine : numpy.ndarray of shape (4, 4)
        Affine matrix written into both output images.
    complex_type : {'real&imaginary', 'magnitude&phase'}, optional
        Which representation to write. Default is ``'real&imaginary'``.

    Raises
    ------
    ValueError
        If ``complex_type`` is not one of the two recognized values.

    Notes
    -----
    Phase is written in radians, wrapped to :math:`(-\\pi, \\pi]` by
    :func:`numpy.angle`. Writing magnitude and phase therefore discards any phase
    unwrapping the caller may have applied.
    """
    data = np.asarray(data, dtype=np.complex128)
    if complex_type == REAL_IMAG:
        first, second = data.real, data.imag
    elif complex_type == MAG_PHASE:
        first, second = np.abs(data), np.angle(data)
    else:
        raise ValueError(f'unknown complex_type {complex_type!r}')

    nb.save(nb.Nifti1Image(first, affine), str(first_path))
    nb.save(nb.Nifti1Image(second, affine), str(second_path))


def complex_file_pair(path, naming=('R_', 'I_')):
    """Derive GIFT's two filenames from a base name, using the write convention.

    Parameters
    ----------
    path : str or pathlib.Path
        Base file name, which must contain an underscore.
    naming : tuple of (str, str), optional
        The two prefixes to prepend. Default is ``('R_', 'I_')`` for real and imaginary;
        use ``('Mag_', 'Phase_')`` for magnitude and phase.

    Returns
    -------
    tuple of (pathlib.Path, pathlib.Path)
        The two derived paths, in the same order as ``naming``.

    Raises
    ------
    ValueError
        If the base name contains no underscore.

    Notes
    -----
    This derives *prefix* names, which is GIFT's **write** convention
    (``WRITE_NAMING_COMPLEX_IMAGES``): ``R_``/``I_`` and ``Mag_``/``Phase_``. GIFT's read
    convention instead defaults to a *suffix* (``_R``/``_I``, ``_CM``/``_CP``) and accepts
    either form; this helper does not generate those. The underscore requirement mirrors
    GIFT, which rejects a base name without one rather than silently accepting it.
    """
    path = Path(path)
    if '_' not in path.name:
        raise ValueError(
            f'complex file names must contain an underscore (GIFT convention): {path.name}'
        )
    return path.with_name(naming[0] + path.name), path.with_name(naming[1] + path.name)
