"""Estimator interface. Every complex ICA estimator returns the same triple."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class EstimatorResult:
    W: np.ndarray  # (N, N) demixing:  S = W @ X
    A: np.ndarray  # (N, N) mixing:    X ~ A @ S
    S: np.ndarray  # (N, T) sources


class Estimator(Protocol):
    def __call__(self, X: np.ndarray, **kwargs) -> EstimatorResult: ...
