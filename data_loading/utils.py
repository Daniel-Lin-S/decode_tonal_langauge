import re
from typing import Dict, List, Optional
import numpy as np


def extract_block_id(filename: str) -> int:
    """
    Extract the integer block ID from a filename containing 'B{block_id}'.
    
    Parameters
    ----------
    filename : str
        The file name (e.g., 'HS25_B1.wav', 'B5_16000.TextGrid')
    
    Returns
    -------
    int
        Extracted block ID as integer
    
    Raises
    ------
    ValueError
        If no 'B<number>' pattern is found.
    """
    match = re.search(r'B(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"No block ID found in filename: {filename}")


def prepare_tone_dynamics(
        tone_dynamic_mapping: Dict[str, List[int]],
        tone_labels: np.ndarray, syllable_labels: np.ndarray
    ) -> np.ndarray:
    """
    Merge tone labels and syllable labels into dynamic features
    using tone_dynamics.

    Parameters
    ----------
    tone_dynamic_mapping: Dict[str, List[int]]
        Dictionary mapping tone IDs to sequences of tone dynamics.
        Each value must have the same length.
    tone_labels: np.ndarray
        Array of tone labels corresponding to the samples.
        Each label should be an integer, and
        should match the keys in tone_dynamic_mapping.
    syllable_labels: np.ndarray
        Array of syllable labels corresponding to the samples.

    Returns
    -------
    np.ndarray
        Array of dynamic features with shape (n_samples, 2, n_dynamics)
        where n_dynamics is the length of the sequences in tone_dynamics.
    """

    if len(tone_labels) != len(syllable_labels):
        raise ValueError(
            "Length of tone labels and syllable labels must match.")
        
    dynamics = []

    for tone, syllable in zip(tone_labels, syllable_labels):
        try:
            tone_dynamic = tone_dynamic_mapping[str(tone)]
        except KeyError:
            raise ValueError(
                f"Tone {str(tone)} not found in tone_dynamic_mapping."
                "Available tones in mapping: "
                f"{list(tone_dynamic_mapping.keys())}")
        syllable_dynamic = [syllable] * len(tone_dynamic)
        dynamic = np.array([syllable_dynamic, tone_dynamic])
        dynamics.append(dynamic)

    dynamics = np.array(dynamics)

    return dynamics


def match_filename(
        file: str, file_format: str,
        kwords: Optional[List[str]]=None,
    ) -> bool:
    """
    Check if the file is of desired form and
    have all the keywords in the filename.

    Parameters
    ----------
    file : str
        The filename to check.
    file_format : str
        The expected file format (e.g., 'wav', 'TextGrid').
    kwords : List[str], optional
        List of keywords to check in the filename.
        If None, no keywords are checked.

    Returns
    -------
    bool
        True if the file matches the criteria, False otherwise.
    """
    # Check file extension
    if not file.endswith(file_format):
        return False

    condition = True

    if kwords is not None:
        for word in kwords:
            if word not in file:
                condition = False

    return condition


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

