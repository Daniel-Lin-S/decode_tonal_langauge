import numpy as np
from argparse import Namespace

def run(data: np.ndarray, params: Namespace) -> np.ndarray:
    """
    z-score normalisation on each channel of the input data.

    Parameters
    ----------
    data : np.ndarray
        Input data array of shape (n_channels, n_timepoints).
    params : object
        Parameters object containing normalisation settings.
        - `preserve_nans`: bool (optional), if True, NaN values in the input data will be preserved in the output.
          Default is True.
    """

    preserve_nans = getattr(params, "preserve_nans", True)

    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    zscored_data = (data - mean) / std
    
    if not preserve_nans:
        zscored_data[np.isnan(zscored_data)] = 0

    return zscored_data
