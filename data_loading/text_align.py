import pandas as pd
import numpy as np
from textgrid import TextGrid
import os
from typing import List, Optional, Dict, Tuple, Sequence

import warnings

from .utils import extract_block_id, match_filename


def handle_textgrids(
        data_dir: str,
        start_offset: float = 0.0,
        end_offset: float = 0.0,
        tier_list: Optional[List[str]]=None,
        blocks: Optional[List[int]]=None,
    ) -> Dict[int, pd.DataFrame]:
    """
    Extracts information from TextGrid files in the specified directory.
    Each TextGrid file must have a naming convention that includes
    a block number at the end,
    e.g. 'example_TextGrid_B1.TextGrid'.
    Each block will only be loaded once.

    Parameters
    ----------
    data_dir : str
        Directory containing TextGrid files.
    start_offset : float, optional
        Offset to subtract from the start time of each interval.
        Defaults to 0.0.
    end_offset : float, optional
        Offset to add to the end time of each interval.
        Defaults to 0.0.
    tier_list : List[str], optional
        List of tier names to extract from the TextGrid files.
        If None, all tiers will be considered.
    blocks : List[int], optional
        List of block numbers to process.
        If None, all blocks will be processed.
    
    Returns
    -------
    Dict[int, pd.DataFrame]
        A dictionary where keys are block numbers and
        values are lists of dictionaries containing the information
        extracted from each TextGrid file. \n
        Each dictionary in the list has the following structure:
        - start : float, start time of the interval (in seconds)
        - end : float, end time of the interval (in seconds)
        - syllable : str, syllable associated with the interval
        - tone : int, tone associated with the interval
    """

    intervals = {}

    for file in os.listdir(data_dir):
        if file.endswith('.TextGrid'):
            block_number = extract_block_id(file)
            
            if blocks is not None and block_number not in blocks:
                continue

            if block_number not in intervals:
                file_path = os.path.join(data_dir, file)
                tg = TextGrid.fromFile(file_path)

                block_data = read_textgrid(
                    tg, start_offset, end_offset, tier_list
                )

                total_len = get_textgrid_time(tg, tier_list)
                print(
                    f'Maximum time for block {block_number}:',
                    total_len, ' s')

                intervals[block_number] = block_data

    return intervals


def read_textgrid(
        tg : TextGrid,
        start_offset: float, end_offset: float,
        tier_list: Optional[List[str]]=None
    ) -> pd.DataFrame:
    """
    Find the intervals with marks starting with a digit
    (indicates true reading times) and extract them into a DataFrame.

    Parameters
    ----------
    tg : TextGrid
        TextGrid object containing the intervals.
    start_offset : float
        Offset to subtract from the start time of each interval,
        in seconds
    end_offset : float
        Offset to add to the end time of each interval,
        in seconds
    tier_list : List[str], optional
        List of tier names to extract from the TextGrid.
        If None, all tiers will be considered.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the extracted intervals.
        Each row represents an interval with the following columns:
        - start : float, start time of the interval (in seconds)
        - end : float, end time of the interval (in seconds)
        - syllable : str, syllable associated with the interval
        - tone : int, tone associated with the interval
    """
    trial_list = []

    tier_names = [tier.name for tier in tg.tiers]
    if tier_list is None:
        tier_list = tier_names

    for tier in tg.tiers:
        if tier.name.lower() in tier_list:
            for interval in tier.intervals:
                if len(interval.mark) == 0:
                    continue

                if interval.mark[0].isdigit():
                    tone = int(interval.mark[0])
                    syllable = interval.mark[1]

                    start = interval.minTime - start_offset
                    end = interval.maxTime + end_offset

                    if trial_list and start < trial_list[-1]['end']:
                        warnings.warn(
                            f"Overlapping intervals detected in tier '{tier.name}' "
                            f"at time {interval.minTime:.2f} for syllable '{syllable}', "
                            f"previous end time was {trial_list[-1]['end']:.2f}. "
                            "Skipping this interval ... "
                        )
                        continue

                    trial_list.append({
                        'start' : np.around(start, decimals=1),
                        'end' : np.around(end, decimals=1),
                        'syllable' : syllable,
                        'tone' : tone
                    })
    
    return pd.DataFrame(trial_list)


def get_textgrid_time(
        tg: TextGrid,
        tier_list: Optional[List[str]]=None
    ) -> float:
    """
    Calculates the total temporal length of the TextGrid file
    based on the specified tiers.

    Parameters
    ----------
    tg : TextGrid
        TextGrid object containing the intervals.
    tier_list : List[str], optional
        List of tier names to consider for length calculation.
        If None, all tiers will be considered.

    Returns
    -------
    float
        The maximum end time of the intervals in the specified tiers.
    """
    if tier_list is None:
        tier_list = [tier.name.lower() for tier in tg.tiers]

    max_time = 0.0

    for tier in tg.tiers:
        if tier.name.lower() in tier_list:
            for interval in tier.intervals:
                if interval.maxTime > max_time:
                    max_time = interval.maxTime

    return max_time


def extract_ecog_audio(
        intervals: Dict[int, pd.DataFrame],
        recording_dir: str,
        syllables: Optional[List[str]] = None,
        length: float = 1.0,
        output_path: Optional[str]=None,
        rest_period: Optional[Sequence[float]] = None,
        recording_format: str = 'npz'
    ) -> Dict[str, np.ndarray]:
    """
    Extracts ECoG and audio samples based on the intervals
    extracted from TextGrid files.

    Parameters
    ----------
    intervals : Dict[int, pd.DataFrame]
        Dictionary where keys are block numbers and
        values are DataFrames containing the intervals.
        Should be output of `handle_textgrids`.
    recording_dir : str
        Directory containing ECoG and audio files. (.npz forms)
        Should have ECoG files with "3052Hz" in the name and
        audio files with "sound" in the name.
        The name of each file must start with "B[block_number]".
        Required keys: `data` for the recording,
        `sf` for sampling frequency.
    syllables : List[str], optional
        List of syllables to map to the intervals.
        Syllable at index i will be encoded as i in the output.
        If not provided, syllable labels will be inferred from the intervals.
    length : float, optional
        Length of the samples to extract, in seconds.
        Defaults to 1.0.
    output_path : str, optional
        Path to save the extracted samples (in .npz forms)
    rest_period : Sequence[float], optional
        (start, end) in seconds for the rest period.
        Extracted for reference.
        If not given, rest samples will not be extracted.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing the extracted samples:
        - 'ecog': ECoG samples, shape (n_samples, n_channels, sample_length)
        - 'ecog_sf': Sampling frequency of ECoG data, scalar
        - 'audio': Audio samples, shape (n_samples, sample_length)
        - 'audio_sf': Sampling frequency of audio data, scalar
        - 'syllable': Syllable labels, shape (n_samples,)
        - 'tone': Tone labels, shape (n_samples,)
        - 'ecog_rest': ECoG rest samples (if rest_period is given),
           shape (n_rest_samples, n_channels, sample_length)
    """
    erp_samples = {}   # event related potentials of ECoG
    if rest_period is not None:
        rest_period = tuple(rest_period)
        ecog_rest_samples = {}
    audio_samples = {}
    syllable_labels = {}
    tone_labels = {}

    if syllables is None:
        syllables = sorted({
            s for df in intervals.values() for s in df.get('syllable', [])
        })

    if syllables:
        print('Syllable mapping used: ', dict(enumerate(syllables)))

    for file in os.listdir(recording_dir):
        if match_filename(file, recording_format, ['ecog']):
            block = extract_block_id(file)
            if block not in intervals:
                continue

            if block in erp_samples:
                warnings.warn(
                    'Found multiple ECoG files for block '
                    f'{block}, skipping file {file}. '
                )
                continue

            file_path = os.path.join(recording_dir, file)

            dataset = np.load(file_path)
            try:
                ecog_data = dataset['data']
            except KeyError:
                raise KeyError(
                    f"Expected key 'data' not found in the npz file {file}. "
                    "Ensure the ECoG data is correctly stored."
                    f"Existing keys {list(dataset.keys())}."
                )
            try:
                ecog_sampling_rate = dataset['sf']
            except:
                raise KeyError(
                    f"Expected key 'sf' not found in the npz file {file}. "
                    "Ensure the sampling frequency is correctly stored."
                    f"Existing keys {list(dataset.keys())}."
                )

            print(
                f'ECoG recording length for block {block}:',
                ecog_data.shape[1] / ecog_sampling_rate, ' s'
            )

            erp_samples[block] = []

            for _, row in intervals[block].iterrows():
                start = int(row['start'] * ecog_sampling_rate)
                end = start + int(length * ecog_sampling_rate)

                if end > ecog_data.shape[1]:
                    raise ValueError(
                        f"Requested sample length exceeds ECoG data length for block {block}. "
                        f"Start: {start}, End: {end}; Data length: {ecog_data.shape[1]}. \n"
                        f"Corresponding interval: {row}. "
                    )

                erp_samples[block].append(ecog_data[:, start:end])
            
            erp_samples[block] = np.array(
                erp_samples[block])  # (n_samples, n_channels, sample_length)
            
            tone_labels[block] = intervals[block]['tone'].to_numpy()

            syllable_categories = pd.Categorical(
                intervals[block]['syllable'], categories=syllables)
            syllable_codes = syllable_categories.codes
            syllable_labels[block] = np.array(syllable_codes)  # (n_samples, )

            if rest_period is not None:
                interval_earliest = intervals[block]['start'].min()

                segment_length = int(length * ecog_sampling_rate)
                rest_start = int(rest_period[0] * ecog_sampling_rate)
                rest_end = int(rest_period[1] * ecog_sampling_rate)

                if rest_period[1] > interval_earliest:
                    warnings.warn(
                        f"Rest period end ({rest_period[1]} s) is after the earliest interval start "
                        f"for block {block} (earliest event time: {interval_earliest} s). "
                        f"Reducing rest period end ..."
                    )
                    rest_end = int(interval_earliest * ecog_sampling_rate)

                # extract rest segments
                ecog_rest_samples[block] = []

                for i in range(rest_start, rest_end, segment_length):
                    if i + segment_length > rest_end:
                        break

                    ecog_rest_samples[block].append(
                        ecog_data[:, i:i + segment_length]
                    )
                
                ecog_rest_samples[block] = np.array(
                    ecog_rest_samples[block])

        elif match_filename(file, recording_format, ['sound']):  # Audio Recording
            block = extract_block_id(file)
            if block not in intervals:
                continue

            if block in audio_samples:
                warnings.warn(
                    'Found multiple audio files for block '
                    f'{block}, skipping file {file}. '
                )
                continue

            file_path = os.path.join(recording_dir, file)

            dataset = np.load(file_path)
            try:
                audio_data = dataset['data']
            except KeyError:
                raise KeyError(
                    f"Expected key 'data' not found in the npz file {file}. "
                    "Ensure the audio data is correctly stored."
                    f"Existing keys {list(dataset.keys())}."
                )
            try:
                audio_sampling_rate = dataset['sf']
            except KeyError:
                raise KeyError(
                    f"Expected key 'sf' not found in the npz file {file}. "
                    "Ensure the sampling frequency is correctly stored."
                    f"Existing keys {list(dataset.keys())}."
                )

            print(
                f'Audio recording length for block {block}:',
                audio_data.shape[1] / audio_sampling_rate, ' s'
            )

            audio_samples[block] = []
            for _, row in intervals[block].iterrows():
                start = int(row['start'] * audio_sampling_rate)
                end = start + int(length * audio_sampling_rate)

                if end > audio_data.shape[1]:
                    raise ValueError(
                        f"Requested sample length exceeds audio data length for block {block}. "
                        f"Start: {start}, End: {end}, Data length: {audio_data.shape[1]}. \n"
                        f"Corresponding interval: {row}. "
                    )

                # assumes the audio data is mono-channel
                audio_samples[block].append(audio_data[0, start:end].flatten())

            audio_samples[block] = np.array(audio_samples[block])   # (n_samples, sample_length)
        
    block_ids = audio_samples.keys()

    if erp_samples.keys() != block_ids:
        raise ValueError(
            "Mismatch between ECoG and audio samples blocks. "
            "Ensure both ECoG and audio files are present for each block."
            f" ECoG blocks found: {erp_samples.keys()},"
            f" Audio blocks found: {block_ids}."
        )

    if len(block_ids) == 0:
        raise ValueError(
            "No valid blocks found in the specified directories."
            f"Blocks in textgrids: {list(intervals.keys())}. "
        )

    # merge blocks
    all_erp_samples = []
    all_audio_samples = []
    all_syllable_labels = []
    all_tone_labels = []

    for block in block_ids:
        all_erp_samples.append(erp_samples[block])
        all_audio_samples.append(audio_samples[block])
        all_syllable_labels.append(syllable_labels[block])
        all_tone_labels.append(tone_labels[block])

    all_erp_samples = np.concatenate(all_erp_samples, axis=0)
    all_audio_samples = np.concatenate(all_audio_samples, axis=0)
    all_syllable_labels = np.concatenate(all_syllable_labels, axis=0)
    all_tone_labels = np.concatenate(all_tone_labels, axis=0)

    min_label = np.min(all_tone_labels)
    if min_label > 0:
        all_tone_labels -= min_label  # make syllable labels start from 0

    if rest_period is not None:
        all_ecog_samples_rest = []
        for block in block_ids:
            all_ecog_samples_rest.append(ecog_rest_samples[block])
        all_ecog_samples_rest = np.concatenate(all_ecog_samples_rest, axis=0)
        print('ECoG rest samples shape:', all_ecog_samples_rest.shape)

    print('ECoG ERP samples shape:', all_erp_samples.shape)
    print('Audio samples shape:', all_audio_samples.shape)
    print('Syllable labels shape:', all_syllable_labels.shape)
    print('Tone labels shape:', all_tone_labels.shape)

    # save as npz file
    output_data = {
        'ecog': all_erp_samples,
        'ecog_sf': ecog_sampling_rate,
        'audio': all_audio_samples,
        'audio_sf': audio_sampling_rate,
        'syllable': all_syllable_labels,
        'tone': all_tone_labels
    }

    if rest_period is not None:
        output_data['ecog_rest'] = all_ecog_samples_rest

    if output_path is not None:
        np.savez(output_path, **output_data)
        print(f"ECoG and audio samples saved to {output_path}")

    return output_data
