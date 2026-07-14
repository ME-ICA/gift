from .base import Estimator, EstimatorResult
from .nc_fastica import nc_fastica

ESTIMATORS = {
    "nc-fastica": nc_fastica,
}


def get_estimator(name):
    """Look up an estimator by name (e.g. 'nc-fastica', 'cebm')."""
    try:
        return ESTIMATORS[name]
    except KeyError:
        raise ValueError(
            f"unknown estimator {name!r}; available: {sorted(ESTIMATORS)}"
        ) from None


__all__ = ["ESTIMATORS", "Estimator", "EstimatorResult", "get_estimator", "nc_fastica"]
