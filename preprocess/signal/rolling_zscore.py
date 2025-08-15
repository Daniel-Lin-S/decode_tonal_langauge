import pandas as pd
import numpy as np


def run(data: np.ndarray, params: object) -> np.ndarray:
    """
    Perform z-score normalization on each channel
    of the input data using a rolling window. (Welford-style method)

    Parameters
    ----------
    data : np.ndarray
        Input data array of shape (n_channels, n_timepoints).
    params : object
        Parameters object containing normalization settings:
        - `window_length`: int, size of the rolling window (in seconds).
          If not provided, defaults to 10.
        - `signal_freq` : int, sampling frequency of the input data.
        - `preserve_nans`: bool (optional), if True,
          NaN values in the input data will be preserved in the output.
          Default is True.

    Returns
    -------
    np.ndarray
        Z-score normalized data of the same shape as `data`.
    """
    window_length = getattr(params, "window_length", 10)
    window_size = int(window_length * params.signal_freq)

    preserve_nans = getattr(params, "preserve_nans", True)

    if window_size <= 1:
        raise ValueError("window_size must be greater than 1.")
    
    df = pd.DataFrame(data.T)

    rolling = df.rolling(
        window=window_size, min_periods=1, center=False
    )
    rolling_mean = rolling.mean()
    rolling_std = rolling.std()

    zscored_df = (df - rolling_mean) / rolling_std

    if not preserve_nans:
        zscored_df.fillna(0, inplace=True)

    return zscored_df.T.to_numpy()
