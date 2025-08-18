import math
from typing import List, Tuple, Union
import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import butter, filtfilt, sosfilt, firwin, lfilter
from argparse import Namespace


def run(data: np.ndarray, params: Namespace) -> np.ndarray:
    """
    Apply a frequency filter to the input data based
    on specified frequency ranges.

    Parameters
    ----------
    data : np.ndarray
        Input data array of shape (n_channels, n_timepoints).
    params : Namespace
        Parameters object containing filtering settings.
        - `bands`: list of dictionaries specifying:
            - `method` : str, filtering method to use, one of
              ['hilbert', 'butter', 'fir'].
            - 'params' : dict (optional), additional parameters for the filter method.
          all the extracted bands will be concatenated as channels.
        - `signal_freq`: int, sampling frequency of the input data.

    Returns
    -------
    np.ndarray
        Filtered data of shape (n_filtered_channels, n_timepoints).
        Where n_filtered_channels depends on the number of bands
        and the filtering method used.
    """

    if "bands" not in params or params.bands is None:
        raise ValueError("bands must be specified in params.")

    all_channels = []
    for freq_config in params.bands:
        method = freq_config.get("method", "hilbert")
        method_params = freq_config.get("params", {})

        if method == 'hilbert':
            if 'freq_ranges' not in method_params:
                raise ValueError(
                    "Hilbert filter requires 'freq_ranges' in params."
                )
            signals = hilbert_filter(
                data, params.signal_freq,
                **method_params
            )
        elif method == 'butter':
            if "freqs" not in method_params:
                raise ValueError(
                    "freqs must be specified when using Butterworth filter. "
                    "Please provide a tuple (low_freq, high_freq) "
                )
            signals = butter_filter(
                data,
                fs=params.signal_freq,
                **method_params
            )
        elif method == 'fir':
            if "order" not in method_params or "center_frequencies" not in method_params:
                raise ValueError(
                    "FIR filter requires 'order' and 'center_frequencies' in params."
                )
            signals = fir_bandpass_filter(
                data,
                fs=params.signal_freq,
                order=method_params["order"],
                center_frequencies=method_params["center_frequencies"]
            )
        all_channels.append(signals)

    filtered = np.concatenate(all_channels, axis=0)

    return filtered


def hilbert_filter(
    data: np.ndarray,
    sampling_rate: int,
    freq_ranges: Union[List[Tuple[float, float]], Tuple[float, float]],
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
    freq_ranges : List[Tuple[float, float]] | Tuple[float, float]
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
    if isinstance(freq_ranges[0], float):
        freq_ranges = [tuple(freq_ranges)]

    C, T = data.shape

    center_freqs = []
    sigma_fs = []

    for freq_range in freq_ranges:
        if len(freq_range) != 2:
            raise ValueError(
                "Each frequency range must be a tuple of (min_freq, max_freq)."
            )
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


def butter_filter(
        data: np.ndarray,
        freqs: Union[Tuple[float, float], float],
        fs: float,
        order: int=4,
        causal: bool=False,
        filter_type: str='bandpass'
    ) -> np.ndarray:
    """
    Apply a bandpass filter to the data.

    Parameters
    ----------
    data : np.ndarray
        The input data to be filtered.
        Of shape (n_channels, n_timepoints) or (n_timepoints,).
    freqs: Tuple[float, float]
        The frequency range (low, high) in Hz for bandpass/bandstop
        or the cutoff frequency in Hz for lowpass/highpass.
    fs : float
        The sampling frequency of the data in Hz.
    order : int, optional
        The order of the Butterworth filter. Default is 4.

    Returns
    -------
    np.ndarray
        The filtered data of the same shape as `data`.
        If `data` is 2D, the output will also be 2D.
        If `data` is 1D, the output will be 1D.
    """
    nyquist = 0.5 * fs
    freqs = np.asarray(freqs, dtype=float)
    normalised_freqs = freqs / nyquist

    if causal:
        sos = butter(order, normalised_freqs, btype=filter_type, output='sos')
        filtered = sosfilt(sos, data, axis=-1)
    else:
        b, a = butter(order, normalised_freqs, btype=filter_type)
        filtered = filtfilt(b, a, data, axis=-1)

    return filtered


def fir_bandpass_filter(
        data: np.ndarray,
        fs: float,
        order: int,
        center_frequencies: List[float]
    ) -> np.ndarray:
    """
    Apply a FIR bandpass filter to the input data
    at given frequencies.

    Parameters
    ----------
    data : np.ndarray
        The input signal (1D or 2D array).
    fs : float
        The sampling frequency of the data in Hz.
    order : int
        The order of the FIR filter (e.g., 390).
    center_frequencies : list of float
        The center frequencies of the bandpass filter in Hz.

    Returns
    -------
    np.ndarray
        The filtered signal of the same shape as `data`.
    """
    nyquist = 0.5 * fs
    filtered = np.zeros_like(data)

    for center_freq in center_frequencies:
        lowcut = center_freq * 0.9
        highcut = center_freq * 1.1

        low = lowcut / nyquist
        high = highcut / nyquist

        fir_coeff = firwin(order + 1, [low, high], pass_zero=False, fs=fs)

        filtered += lfilter(fir_coeff, 1.0, data, axis=-1)

    filtered /= len(center_frequencies)

    return filtered


