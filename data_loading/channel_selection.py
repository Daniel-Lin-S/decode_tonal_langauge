import numpy as np
from scipy.stats import f_oneway
from typing import Dict, Tuple, List, Mapping


def test_discriminative_power(
    data: Mapping[str, np.ndarray],
    label_name: str,
    recording_name: str='ecog'
) -> Dict[str, np.ndarray]:
    """
    Test the discriminative power of each recording channel on a given label
    using one-way ANOVA.

    Parameters
    ----------
    data : Mapping[str, np.ndarray]
        Any dictionary-like structure containing the recordings and labels.
        The recording must have shape (n_samples, n_channels, n_timepoints).
        And the labels must have shape (n_samples, 1) or (1, n_samples)
        with values being integers (e.g., categorical labels).
    label_name : str
        Name of the label to test (e.g., 'syllable' or 'tone').
    recording_name : str, optional
        Name of the event-related potential recordings
        should have the same length as lables.
        (default is 'ecog').

    Returns
    -------
    dict
        Dictionary containing F-statistics and p-values for each channel
        at each timepoint.
        Format: {'f_stat': np.ndarray, 'p_value': np.ndarray}
        Both arrays have shape (n_channels, n_timepoints).
    """
    try:
        series = data[recording_name]
    except KeyError:
        raise KeyError(
            f"Recording '{recording_name}' not found in data."
            f"Available keys: {list(data.keys())}"
        )

    if series.ndim != 3:
        raise ValueError(
            f"Recording '{recording_name}' must be a 3D array "
            "(n_samples, n_channels, n_timepoints)."
        )

    try:
        labels = data[label_name].squeeze()  #  (n_samples,)
    except KeyError:
        raise KeyError(
            f"Labels '{label_name}' not found in data."
            f"Available keys: {list(data.keys())}"
        )

    if labels.ndim != 1:
        raise ValueError(
            f"Labels '{label_name}' must be a 1D array "
            "(n_samples,) or 2D array with shape (1, n_samples)"
            " or (n_samples, 1)."
        )

    if labels.shape[0] != series.shape[0]:
        raise ValueError(
            f"Number of samples in '{label_name}' ({labels.shape[0]}) "
            "does not match number of samples in "
            f"'{recording_name}' ({series.shape[0]})."
        )

    # Ensure labels are integers (e.g., categorical labels)
    unique_labels = np.unique(labels)
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"Labels for '{label_name}' must be integers.")

    n_channels, n_timepoints = series.shape[1:]

    f_stat = np.zeros((n_channels, n_timepoints))
    p_value = np.zeros((n_channels, n_timepoints))

    # Perform one-way ANOVA for each channel
    for channel_idx in range(n_channels):
        channel_data = series[:, channel_idx, :]  # (n_samples, n_timepoints)

        grouped_data = [
            channel_data[labels == label, :] for label in unique_labels]

        f_result = f_oneway(*grouped_data)
        f_stat[channel_idx] = f_result.statistic   # (n_timepoints,)
        p_value[channel_idx] = f_result.pvalue   # (n_timepoints,)

    return {'f_stat': f_stat, 'p_value': p_value}


def get_max_length(indices: np.ndarray) -> int:
    """
    Get the maximum length of consecutive indices in the
    array.

    Parameters
    ----------
    indices : np.ndarray
        Sorted array of indices where the condition is met.

    Returns
    -------
    int
        Maximum length of consecutive indices.
    """
    segments = []
    current = [indices[0]]

    for idx in indices[1:]:
        if idx == current[-1] + 1:
            current.append(idx)
        else:
            segments.append(current)
            current = [idx]

    segments.append(current)
    return max(len(segment) for segment in segments)
    

def find_significant_channels(
        f_test_results: Dict[str, np.ndarray],
        pvalue_threshold: float = 0.05,
        consecutive_length_threshold: int = 10
    ) -> list:
    """
    Find channels with significant discriminative power based on p-values.

    Parameters
    ----------
    f_test_results : dict
        Dictionary containing F-statistics and p-values for each channel
        at each timepoint, as returned by `test_discriminative_power`.
    threshold : float, optional
        Significance threshold for p-values (default is 0.05).
    """
    p_values = f_test_results['p_value']
    significant_channels = []

    for ch in range(p_values.shape[0]):
        significant_indices = np.where(p_values[ch, :] < pvalue_threshold)[0]

        if len(significant_indices) > 0:
            max_length = get_max_length(significant_indices)
            if max_length > consecutive_length_threshold:
                significant_channels.append(ch)
        
    return significant_channels


def find_active_channels(
    data: Mapping[str, np.ndarray],
    rest_recording_name: str = 'ecog_rest',
    erp_recording_name: str = 'ecog',
    p_val_threshold: float = 0.05,
    length_threshold: int = 10
) -> Tuple[List[int], List[int]]:
    """
    Find channels that are significantly active during the ERP recording
    by comparing it to the rest recording using one-way ANOVA.

    Parameters
    ----------
    data : Mapping[str, np.ndarray]
        Any dictionary-like structure that contains
        the recordings during the rest period and the ERP recording,
        both should be numpy arrays with shape
        (n_samples, n_channels, n_timepoints).
        Number of samples can be different for each recording,
        but the number of channels and timepoints must match.
    rest_recording_name : str, optional
        Name of the rest recording (default is 'ecog_rest').
    erp_recording_name : str, optional
        Name of the ERP recording (default is 'ecog').
        Note: each sample of the ERP recording and the rest recording
    p_val_threshold : float, optional
        Significance threshold for p-values (default is 0.05).
    length_threshold : int, optional
        Minimum length of consecutive significant timepoints
        to consider a channel active (default is 10).

    Returns
    -------
    active_channels : List[int]
        List of indices of channels that are significantly active.
    max_lengths : List[int]
        List of maximum lengths of consecutive significant timepoints
        for each active channel.
    """

    try:
        rest_samples = data[rest_recording_name]
    except KeyError:
        raise KeyError(
            f"Recording '{rest_recording_name}' not found in data."
            f"Available keys: {list(data.keys())}"
        )
    
    try:
        erp_samples = data[erp_recording_name]
    except KeyError:
        raise KeyError(
            f"Recording '{erp_recording_name}' not found in data."
            f"Available keys: {list(data.keys())}"
        )

    if erp_samples.shape[1:2] != rest_samples.shape[1:2]:
        raise ValueError(
            f"Shape mismatch between '{erp_recording_name}' "
            f"and '{rest_recording_name}': "
            f"{erp_samples.shape[1:2]} vs {rest_samples.shape[1:2]}."
        )

    # perform one-way ANOVA for each channel
    n_channels = rest_samples.shape[1]
    active_channels = []
    max_lengths = []

    for ch in range(n_channels):
        rest_data = rest_samples[:, ch, :]
        erp_data = erp_samples[:, ch, :]

        result = f_oneway(rest_data, erp_data)

        p_vals = result.pvalue

        significant_points = np.where(p_vals < p_val_threshold)[0]

        if len(significant_points) == 0:
            continue

        max_len = get_max_length(np.where(p_vals < p_val_threshold)[0])

        if max_len > length_threshold:
            active_channels.append(ch)
            max_lengths.append(max_len)
    
    return active_channels, max_lengths
