"""
Extracts ECoG and audio samples by aligning with intervals.
"""

import pandas as pd
import numpy as np
import os
from typing import List, Optional, Dict, Sequence
import warnings
from argparse import Namespace
import matplotlib.pyplot as plt

from .utils import extract_block_id, match_filename


def get_samples(
        intervals: Dict[int, pd.DataFrame],
        params: Namespace
    ) -> Dict[str, np.ndarray]:
    """
    Extracts ECoG and audio samples based on the intervals.

    Parameters
    ----------
    intervals: Dict[int, pd.DataFrame]
        A dictionary where keys are block numbers and
        values are DataFrames containing the extracted intervals
        with columns:
        - start : float, start time of the interval (in seconds)
        - end : float, end time of the interval (in seconds)
        - syllable : str, syllable associated with the interval
        - tone : int, tone associated with the interval
    data_dir : str
        Directory containing ECoG and audio files. (.npz forms)
        Should have ECoG files with "3052Hz" in the name and
        audio files with "sound" in the name.
        The name of each file must start with "B[block_number]".
        Required keys: `data` for the recording,
        `sf` for sampling frequency.
    output_path : str, optional
        Path to save the extracted samples (in .npz forms)
        If None, samples will not be saved.
    params : Namespace 
        Parameters for the extraction process.
        Should contain:
        - sample_length : float, length of each sample in seconds
        - recording_format : str, format of the recording files,
          e.g. '.npz'
        - syllable_identifiers : List[str], optional
            List of syllable identifiers to use for labeling.
            If None, all syllables found in the intervals will be used,
            and a default number mapping is given.
        - rest_period : Sequence[float], optional
            Rest period to extract ECoG rest samples.
            Should be a tuple of two floats (start, end) in seconds.
            If None, no rest samples will be extracted.

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
    length = getattr(params, 'sample_length', 1.0)
    recording_format = getattr(params, 'recording_format', '.npz')
    syllables: List[str] = getattr(params, 'syllable_identifiers', None)
    rest_period: Sequence[float] = getattr(params, 'rest_period', None)

    generate_figures(
        intervals, params.data_dir,
        figure_dir=os.path.join(os.path.dirname(params.output_path), 'figures'),
        subject_id=params.subject_id
    )

    erp_samples = {}
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

    print('Syllable mapping used: ', dict(enumerate(syllables)), flush=True)

    for file in os.listdir(params.data_dir):
        if match_filename(file, recording_format, ['ecog']):
            block = extract_block_id(file)

            if block not in intervals:
                continue

            if 'start' not in intervals[block].columns or \
                'end' not in intervals[block].columns or \
                'syllable' not in intervals[block].columns or \
                'tone' not in intervals[block].columns:
                raise ValueError(
                    f"Intervals for block {block} do not contain all required columns: "
                    "'start', 'end', 'syllable', 'tone'. "
                    f"Available columns: {list(intervals[block].columns)}."
                )

            if block in erp_samples:
                warnings.warn(
                    'Found multiple ECoG files for block '
                    f'{block}, skipping file {file}. '
                )
                continue

            file_path = os.path.join(params.data_dir, file)

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
                ecog_data.shape[1] / ecog_sampling_rate, ' s',
                flush=True
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

        elif match_filename(file, recording_format, ['audio']):  # Audio Recording
            block = extract_block_id(file)

            if block not in intervals:
                continue

            if block in audio_samples:
                warnings.warn(
                    'Found multiple audio files for block '
                    f'{block}, skipping file {file}. '
                )
                continue

            file_path = os.path.join(params.data_dir, file)

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
                audio_data.shape[1] / audio_sampling_rate, ' s',
                flush=True
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
        
    audio_block_ids = audio_samples.keys()
    ecog_block_ids = erp_samples.keys()

    if audio_block_ids != ecog_block_ids:
        raise ValueError(
            "Mismatch between ECoG and audio samples blocks. "
            "Ensure both ECoG and audio files are present for each block."
            f" ECoG blocks found: {ecog_block_ids},"
            f" Audio blocks found: {audio_block_ids}."
        )

    if len(audio_block_ids) == 0 or len(ecog_block_ids) == 0:
        raise ValueError(
            "No ECoG or audio samples found. "
            f"Blocks in intervals: {intervals.keys()}. "
            f"Files in the directory: {os.listdir(params.data_dir)}"
        )

    # merge blocks
    all_erp_samples = []
    all_audio_samples = []
    all_syllable_labels = []
    all_tone_labels = []

    for block in ecog_block_ids:
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
        for block in ecog_block_ids:
            all_ecog_samples_rest.append(ecog_rest_samples[block])
        all_ecog_samples_rest = np.concatenate(all_ecog_samples_rest, axis=0)
        print('ECoG rest samples shape:', all_ecog_samples_rest.shape, flush=True)

    print('ECoG ERP samples shape:', all_erp_samples.shape, flush=True)
    print('Audio samples shape:', all_audio_samples.shape, flush=True)
    print('Syllable labels collected:', np.unique(all_syllable_labels), flush=True)
    print('Tone labels collected:', np.unique(all_tone_labels), flush=True)

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

    if params.output_path is not None:
        np.savez(params.output_path, **output_data)
        print(f"ECoG and audio samples saved to {params.output_path}", flush=True)

    return output_data


def generate_figures(
        intervals: Dict[int, pd.DataFrame],
        data_dir: str,
        figure_dir: str,
        subject_id: str
    ):
    if len(intervals) == 0:
        warnings.warn(
            'No intervals found in the annotation files. '
            'Will not be able to generate figures.'
        )
        return
            
    for block_id, block_df in intervals.items():
        if not block_df.empty:
            events = block_df.to_dict('records')
            sampled_events = _sample_consecutive_events(events, num_events=3)

            data_path = os.path.join(data_dir, f"B{block_id}_ecog.npz")

            if os.path.exists(data_path):
                ecog = np.load(data_path)
                signal = ecog['data']
                sf = int(ecog['sf'])
                channels = np.random.choice(
                    signal.shape[0], size=5, replace=False)
                subject_fig_dir = os.path.join(figure_dir, f'subject_{subject_id}')
                os.makedirs(subject_fig_dir, exist_ok=True)

                plot_ecog_events(
                    signal, sf, sampled_events, channels,
                    subject_id, block_id, subject_fig_dir
                )
            else:
                warnings.warn(
                    f"ECoG data for block {block_id} not found at {data_path}. "
                    "Skipping figure generation for this block."
                    f"Files available: {os.listdir(data_dir)}"
                )


def _sample_consecutive_events(events, num_events):
    events = sorted(events, key=lambda x: x['start'])

    if len(events) > num_events:
        start_idx = np.random.randint(0, len(events) - num_events + 1)
        return events[start_idx:start_idx + num_events]
    else:
        return events

def plot_ecog_events(
    signal: np.ndarray,
    sf: int,
    events: list,
    channels: list,
    subject_id: str,
    block_id: str,
    fig_dir: str
) -> None:
    """
    Plot ECoG signal for multiple channels, with each channel occupying a subplot.
    Highlight events and include inter-event signals.

    Args:
        signal (np.ndarray): ECoG signal (channels x timepoints).
        sf (int): Sampling frequency of the signal.
        events (list): List of event intervals, each as a dict with 'start' and 'end'.
        channels (list): List of channel indices to plot.
        subject_id (str): Subject ID for labeling.
        block_id (str): Block ID for labeling.
        fig_dir (str): Directory to save the figure.
    """
    os.makedirs(fig_dir, exist_ok=True)

    start_time = max(min(event['start'] for event in events) - 0.5, 0)
    end_time = max(event['end'] for event in events) + 0.5
    start_idx = int(start_time * sf)
    end_idx = int(end_time * sf)
    time = np.arange(start_idx, end_idx) / sf

    fig, axes = plt.subplots(len(channels), 1, figsize=(12, 4 * len(channels)), sharex=True)
    if len(channels) == 1:
        axes = [axes]

    for ax, ch_idx in zip(axes, channels):
        ax.plot(
            time,
            signal[ch_idx, start_idx:end_idx],
            label=f'Offset',
            color='blue',
            alpha=0.7
        )

        for i, event in enumerate(events):
            event_start_idx = int(event['start'] * sf)
            event_end_idx = int(event['end'] * sf)
            event_time = np.arange(event_start_idx, event_end_idx) / sf

            ax.plot(
                event_time,
                signal[ch_idx, event_start_idx:event_end_idx],
                label=f'Onset' if i == 0 else None,
                color='orange'
            )
            ax.axvline(
                event['start'], color='g',
                linestyle='--', alpha=0.7,
                label='Event Start' if i == 0 else None
            )
            ax.axvline(
                event['end'], color='r',
                linestyle='--', alpha=0.7,
                label='Event End' if i == 0 else None
            )

        ax.set_title(f'Channel {ch_idx}', fontsize=18)
        ax.set_ylabel('Amplitude', fontsize=16)

        ax.legend(
            fontsize=14, loc='upper right',
            bbox_to_anchor=(1.2, 1),
            borderaxespad=0.
        )

    axes[-1].set_xlabel('Time (s)', fontsize=16)
    fig.suptitle(f'Subject {subject_id} Block {block_id}', fontsize=20)
    fig.tight_layout()

    fig.subplots_adjust(top=0.93)

    fig_path = os.path.join(fig_dir, f'block_{block_id}_events.png')
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
