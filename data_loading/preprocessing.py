import math
import numpy as np
from scipy.fft import fft, ifft
from typing import Tuple, List
from scipy import signal


def hilbert_filter(
    data: np.ndarray,
    sampling_rate: int,
    freq_ranges: List[Tuple[float, float]],
    f0: float = 0.018,
    octspace: float = 1/7,
    filterbank_bias: float = math.log10(0.39),
    filterbank_slope: float = 0.5,
    envelope: bool = True
) -> np.ndarray:
    """
    Apply a Gaussian Hilbert filter bank to multichannel data.

    Parameters
    ----------
    data : np.ndarray
        Input shape (n_channels, n_timepoints)
    sampling_rate : int
        Data sampling rate (Hz)
    freq_ranges : List[Tuple[float, float]]
        List of frequency ranges to filter. \n
        Each range is a tuple (min_freq, max_freq).
        Can also give a single tuple for one band.
    f0 : float
        Base frequency
    octspace : float
        Spacing in octaves
    filterbank_bias : float
        Filter bank bias (log scale)
    filterbank_slope : float
        Filter bank slope
    envelope : bool
        If True, return the Hilbert-transformed envelope signals.
        If False, return the real part of the Hilbert transform.

    Returns
    -------
    filtered_signal : np.ndarray
        Hilbert-transformed envelope signals,
        shape (n_channels, n_timepoints)
    """
    if isinstance(freq_ranges, tuple):
        freq_ranges = [freq_ranges]

    C, T = data.shape

    center_freqs = []
    sigma_fs = []

    for freq_range in freq_ranges:
        min_freq = freq_range[0] if freq_range else 0
        max_freq = freq_range[1] if freq_range else sampling_rate // 2
        max_oct = math.log2(max_freq / f0)

        # Generate center frequencies
        f = f0
        while math.log2(f / f0) < max_oct:
            if f >= min_freq:
                center_freqs.append(f)
                sigma_fs.append(10 ** (
                    filterbank_bias + filterbank_slope * math.log10(f))
                )
            f = f * (2 ** octspace)

    center_freqs = np.array(center_freqs)
    sigma_fs = np.array(sigma_fs) * np.sqrt(2)
    n_banks = len(center_freqs)

    # FFT frequency bins
    freqs = np.fft.fftfreq(T, d=1. / sampling_rate)

    # Hilbert transform multiplier
    hilbert_mult = np.zeros(T)
    if T % 2 == 0:
        hilbert_mult[0] = 1
        hilbert_mult[1:T//2] = 2
        hilbert_mult[T//2] = 1
    else:
        hilbert_mult[0] = 1
        hilbert_mult[1:(T + 1)//2] = 2

    data_fft = fft(data, axis=1)

    # Filter bank application
    filtered_signal = np.zeros((C, T, n_banks))
    for i, (f_c, s_f) in enumerate(zip(center_freqs, sigma_fs)):
        H = np.exp(-0.5 * ((freqs - f_c) / s_f) ** 2)
        H[0] = 0  # remove DC

        filter_kernel = H * hilbert_mult
        for ch in range(C):
            signal = ifft(data_fft[ch] * filter_kernel)
            if envelope:
                filtered_signal[ch, :, i] = np.abs(signal)
            else:
                filtered_signal[ch, :, i] = signal.real

    # Mean envelope over bands
    return filtered_signal.mean(axis=2)


def downsample(
        data: np.ndarray, source_freq: int, target_freq: int
    ) -> np.ndarray:
    """
    Downsample the input data from source_freq to target_freq.

    Parameters
    ----------
    data : np.ndarray
        Input data array of shape (n_channels, n_timepoints).
    source_freq : int
        Original sampling frequency (in Hz) of the data.
    target_freq : int
        Desired sampling frequency (in Hz) after downsampling.

    Returns
    -------
    np.ndarray
        Downsampled data of shape (n_channels, n_new_timepoints).
        Which has samplinng rate of target_freq.
    """

    n_samples = math.ceil(
        int(data.shape[1] * (target_freq / source_freq)))
    data_ds = signal.resample(data, n_samples, axis=1)

    return data_ds


def zscore(data: np.ndarray, preserve_nans: bool=True) -> np.ndarray:
    """
    Apply z-score normalisation to the input data.
    Normalisation applied to each channel independently.

    Parameters
    ----------
    data : np.ndarray
        Input data array of shape (n_channels, n_timepoints).
    preserve_nans : bool
        If True, NaN values in the input data will be preserved in the output.
        If False, NaN values will be replaced with zeros.

    Returns
    -------
    np.ndarray
        Z-scored data of the same shape as `data`.
    """
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    zscored_data = (data - mean) / std
    
    if not preserve_nans:
        zscored_data[np.isnan(zscored_data)] = 0

    return zscored_data


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
