import numpy as np


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
    significant_channels
        List of indices of channels that are significantly discriminative.
    max_lengths
        List of maximum lengths of
        consecutive significant timepoints for each channel.
    """
    # Bonferroni correction for multiple comparisons
    pvalue_threshold /= p_values.shape[1]

    significant_channels = []
    max_lengths = []

    for ch in range(p_values.shape[0]):
        significant_indices = np.where(p_values[ch, :] < pvalue_threshold)[0]

        if len(significant_indices) > 0:
            max_length = get_max_length(significant_indices)
            if max_length > length_threshold:
                significant_channels.append(ch)

    return significant_channels, max_lengths
