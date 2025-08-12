from argparse import Namespace
import numpy as np


def run(data: np.ndarray, params: Namespace) -> np.ndarray:
    """
    Apply Common Average Referencing (CAR) to the input data.

    Parameters
    ----------
    data : np.ndarray
        Input data array of shape (n_channels, n_timepoints).
    params : Namespace
        It should have the attributes:
        - `exclude_channels`: List of channel indices (integers) to
          exclude from the CAR computation (optional).

    Returns
    -------
    np.ndarray
        CAR-referenced data of the same shape as `data`.
    """
    if not hasattr(params, 'exclude_channels'):
        params.exclude_channels = []  # Default to no excluded channels

    exclude_channels = params.exclude_channels

    if not isinstance(exclude_channels, list):
        raise ValueError("exclude_channels must be a list of integers.")

    if any(ch < 0 or ch >= data.shape[0] for ch in exclude_channels):
        raise ValueError("exclude_channels contains invalid channel indices.")

    include_mask = np.ones(data.shape[0], dtype=bool)
    include_mask[exclude_channels] = False

    common_average = np.mean(data[include_mask, :], axis=0, keepdims=True)

    car_data = data - common_average

    return car_data
