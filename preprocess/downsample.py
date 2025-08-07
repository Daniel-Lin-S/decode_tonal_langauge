from scipy.signal import resample
from argparse import Namespace
import numpy as np


def run(data: np.ndarray, params: Namespace) -> np.ndarray:
    """
    Downsample the input data to a target frequency specified in params.

    Parameters
    ----------
    data : np.ndarray
        Input data array of shape (n_channels, n_timepoints).
    params : Namespace
        Parameters of the downsampling settings
        - `downsample_freq`: int (optional), target frequency to downsample the data to.
          Default is 400 Hz.
        - `signal_freq`: int, source frequency of the input data.
    """

    target_freq = getattr(params, "downsample_freq", 400)

    factor = target_freq / params.signal_freq
    n_samples = int(data.shape[1] * factor)
    data_ds = resample(data, n_samples, axis=1)

    params.signal_freq = target_freq

    return data_ds
