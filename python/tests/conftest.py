import sys
from pathlib import Path

import pytest
from scipy.io import loadmat

# make tests/oracle.py importable as `oracle`
sys.path.insert(0, str(Path(__file__).parent))

FIXTURES = Path(__file__).resolve().parents[2] / 'complex_ica_fixtures'


@pytest.fixture(scope='session')
def fixtures_dir():
    assert FIXTURES.is_dir(), f'MATLAB oracle fixtures not found at {FIXTURES}'
    return FIXTURES


@pytest.fixture(scope='session')
def sources(fixtures_dir):
    """Ground truth from MATLAB: cX = A @ cS."""
    d = loadmat(fixtures_dir / 'complex_sources.mat')
    return {'cS': d['cS'], 'A': d['A'], 'cX': d['cX']}
