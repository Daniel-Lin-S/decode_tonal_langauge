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
        p_values : np.ndarray,
        pvalue_threshold: float = 0.05,
        length_threshold: int = 10
    ) -> list:
    """
    Find channels with significant discriminative power based on p-values.
    A channel is considered significant if it has at least one timepoint
    with a p-value below the threshold and the maximum length of
    consecutive significant timepoints exceeds the specified threshold.

    Parameters
    ----------
    p_values : np.ndarray
        2D array of p-values with shape (n_channels, n_timepoints).
    pvalue_threshold : float, optional
        Significance threshold for p-values (default is 0.05).
    length_threshold : int, optional
        Minimum length of consecutive significant timepoints
        to consider a channel discriminative (default is 10).

    Return
    -------
    list
        List of indices of channels that are significantly discriminative.
    """
    # Bonferroni correction for multiple comparisons
    pvalue_threshold /= p_values.shape[1]  # Adjust for number of timepoints

    significant_channels = []

    for ch in range(p_values.shape[0]):
        significant_indices = np.where(p_values[ch, :] < pvalue_threshold)[0]

        if len(significant_indices) > 0:
            max_length = get_max_length(significant_indices)
            if max_length > length_threshold:
                significant_channels.append(ch)
        
    return significant_channels


def select_non_discriminative_channels(
        channel_selections: dict,
        discriminative_keys: List[str]
    ) -> list:
    """
    Select channels that are not discriminative based on the provided
    channel selections.
    Channel selections must have the following structure:
    - 'active_channels': List[int], the indices of active channels.
    - for each discriminative key (e.g., 'syllable', 'tone'):
        List[int], the indices of channels that are discriminative for that key.

    Parameters
    ----------
    channel_selections : dict
        Dictionary containing channel selections.
        Must contain 'active_channels' and keys for discriminative channels.
    discriminative_keys : List[str]
        List of keys for discriminative channels to exclude.
    
    Returns
    -------
    list
        List of indices of non-discriminative channels.
    """

    non_discriminative_channels = set(channel_selections['active_channels'])
    discriminative_channels = set()
    for label in discriminative_keys:
        discriminative_channels.update(channel_selections[label])

    non_discriminative_channels = list(
        non_discriminative_channels - discriminative_channels
    )

    non_discriminative_channels.sort()
    
    return non_discriminative_channels
