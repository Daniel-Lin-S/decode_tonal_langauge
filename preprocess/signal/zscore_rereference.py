from argparse import Namespace
import numpy as np
from typing import Tuple


def run(data: np.ndarray, params: Namespace) -> np.ndarray:
    """
    Rereference the input data using a specified reference interval.

    Parameters
    ----------
    data : np.ndarray
        Input data array of shape (n_channels, n_timepoints).
    params : Namespace
        It should have the attributes:
        - `rereference_interval`: A tuple of (start, end) in seconds.
        - `signal_freq`: Sampling frequency of the signal in Hz.
    """
    if not hasattr(params, 'rereference_interval') or not hasattr(params, 'signal_freq'):
        raise ValueError(
            "params must have 'rereference_interval' and 'signal_freq' attributes."
        )

    start, end = params.rereference_interval
    start_sample = int(start * params.signal_freq)
    end_sample = int(end * params.signal_freq)

    rereferenced_data = rereference(data, (start_sample, end_sample))

    return rereferenced_data


def rereference(
    data: np.ndarray,
    reference_time: Tuple[float, float]
) -> np.ndarray:
    """
    Rereference the input data to a specified reference time.
    This function computes the mean of the data within the
    specified reference time range and subtracts it from the data.

    Parameters
    ----------
    data : np.ndarray
        Input data array of shape (n_channels, n_timepoints).
    reference_time : Tuple[float, float]
        A tuple specifying the start and end indices for the reference time.
        The indices should be within the bounds of the data's time dimension.

    Returns
    -------
    np.ndarray
        Rereferenced data of the same shape as `data`.
    """
    try:
        start, end = reference_time
    except ValueError:
        raise ValueError("reference_time must be a tuple of (start, end)")

    if start < 0 or end > data.shape[1]:
        raise ValueError("Reference time indices are out of bounds.")
    
    if start >= end:
        raise ValueError("Start time must be less than end time.")
    
    ref_mean = np.mean(data[:, start:end], axis=1, keepdims=True)  # (n_channels, 1)
    ref_std = np.std(data[:, start:end], axis=1, keepdims=True)  # (n_channels, 1)
    rereferenced_data = (data - ref_mean) / ref_std

    return rereferenced_data
