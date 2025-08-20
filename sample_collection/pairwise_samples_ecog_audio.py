"""Extract speech onset and pre-onset ECoG and audio samples."""

import pandas as pd
import numpy as np
import os
from typing import Dict
from argparse import Namespace
import warnings

from .utils import extract_block_id, match_filename
from .samples_ecog_audio import generate_figures


def get_samples(intervals: Dict[int, pd.DataFrame], params: Namespace) -> Dict[str, np.ndarray]:
    """Extract paired speech-onset and pre-onset samples.

    Parameters
    ----------
    intervals : Dict[int, pd.DataFrame]
        Mapping from block id to DataFrame of intervals with columns
        'start' and 'end'.
    params : Namespace
        Expected fields:
            - sample_length: float, length of onset sample in seconds.
            - max_pre_length: float, maximum length for pre-onset sample in seconds.
            - recording_format: str, file extension of recordings (default '.npz').
            - onset_position: str, either 'start' or 'end' indicating where
              speech onset is referenced in the interval (default 'start').
            - output_path: optional, path to save resulting npz.
            - data_dir: directory containing processed recordings.
    Returns
    -------
    Dict[str, np.ndarray]
        Contains keys 'ecog_onset', 'ecog_pre', 'ecog_sf',
        'audio_onset', 'audio_pre', 'audio_sf',
        'syllable', 'tone'.
    """
    sample_length = getattr(params, 'sample_length', 1.0)
    pre_length = getattr(params, 'max_pre_length', sample_length)
    recording_format = getattr(params, 'recording_format', '.npz')
    onset_pos = getattr(params, 'onset_position', 'start')

    generate_figures(
        intervals,
        params.data_dir,
        figure_dir=os.path.join(os.path.dirname(params.output_path), 'figures'),
        subject_id=params.subject_id,
    )

    ecog_onset = {}
    ecog_pre = {}
    audio_onset = {}
    audio_pre = {}
    syllable_labels = {}
    tone_labels = {}

    syllables = getattr(params, 'syllable_identifiers', None)
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
            if block in ecog_onset:
                warnings.warn(
                    f'Found multiple ECoG files for block {block}, skipping {file}.')
                continue
            file_path = os.path.join(params.data_dir, file)
            dataset = np.load(file_path)
            try:
                ecog_data = dataset['data']
                ecog_sf = int(dataset['sf'])
            except KeyError:
                raise KeyError(
                    f"Expected keys 'data' and 'sf' not found in {file}. "
                    f"Existing keys {list(dataset.keys())}.")

            block_intervals = intervals[block]
            required_cols = {'start', 'end', 'syllable', 'tone'}
            if not required_cols.issubset(block_intervals.columns):
                raise ValueError(
                    f"Intervals for block {block} must contain columns {required_cols}. "
                    f"Available: {list(block_intervals.columns)}")
            block_intervals = block_intervals.sort_values('start').reset_index(drop=True)
            ecog_onset[block] = []
            ecog_pre[block] = []
            prev_end = 0.0
            for _, row in block_intervals.iterrows():
                onset_time = row[onset_pos]
                start_idx = int(onset_time * ecog_sf)
                length_idx = int(sample_length * ecog_sf)
                end_idx = start_idx + length_idx
                if end_idx > ecog_data.shape[1]:
                    raise ValueError(
                        f"Requested sample exceeds ECoG data for block {block}.")
                ecog_onset[block].append(ecog_data[:, start_idx:end_idx])

                pre_end = start_idx
                pre_start_time = max(onset_time - pre_length, prev_end)
                pre_start_idx = int(pre_start_time * ecog_sf)
                pre_seg = ecog_data[:, pre_start_idx:pre_end]
                desired = int(pre_length * ecog_sf)
                if pre_seg.shape[1] < desired:
                    pad = np.zeros((ecog_data.shape[0], desired - pre_seg.shape[1]))
                    pre_seg = np.concatenate([pad, pre_seg], axis=1)
                ecog_pre[block].append(pre_seg[:, -desired:])
                prev_end = row['end']
            ecog_onset[block] = np.array(ecog_onset[block])
            ecog_pre[block] = np.array(ecog_pre[block])
            tone_labels[block] = block_intervals['tone'].to_numpy()
            syllable_categories = pd.Categorical(
                block_intervals['syllable'], categories=syllables)
            syllable_labels[block] = syllable_categories.codes

        elif match_filename(file, recording_format, ['audio']):
            block = extract_block_id(file)
            if block not in intervals:
                continue
            if block in audio_onset:
                warnings.warn(
                    f'Found multiple audio files for block {block}, skipping {file}.')
                continue
            file_path = os.path.join(params.data_dir, file)
            dataset = np.load(file_path)
            try:
                audio_data = dataset['data']
                audio_sf = int(dataset['sf'])
            except KeyError:
                raise KeyError(
                    f"Expected keys 'data' and 'sf' not found in {file}. "
                    f"Existing keys {list(dataset.keys())}.")

            block_intervals = intervals[block]
            required_cols = {'start', 'end', 'syllable', 'tone'}
            if not required_cols.issubset(block_intervals.columns):
                raise ValueError(
                    f"Intervals for block {block} must contain columns {required_cols}. "
                    f"Available: {list(block_intervals.columns)}")
            block_intervals = block_intervals.sort_values('start').reset_index(drop=True)
            audio_onset[block] = []
            audio_pre[block] = []
            prev_end = 0.0
            for _, row in block_intervals.iterrows():
                onset_time = row[onset_pos]
                start_idx = int(onset_time * audio_sf)
                length_idx = int(sample_length * audio_sf)
                end_idx = start_idx + length_idx
                if end_idx > audio_data.shape[1]:
                    raise ValueError(
                        f"Requested sample exceeds audio data for block {block}.")
                audio_onset[block].append(audio_data[0, start_idx:end_idx].flatten())

                pre_end = start_idx
                pre_start_time = max(onset_time - pre_length, prev_end)
                pre_start_idx = int(pre_start_time * audio_sf)
                pre_seg = audio_data[0, pre_start_idx:pre_end].flatten()
                desired = int(pre_length * audio_sf)
                if pre_seg.shape[0] < desired:
                    pad = np.zeros(desired - pre_seg.shape[0])
                    pre_seg = np.concatenate([pad, pre_seg])
                audio_pre[block].append(pre_seg[-desired:])
                prev_end = row['end']
            audio_onset[block] = np.array(audio_onset[block])
            audio_pre[block] = np.array(audio_pre[block])

    block_ids = ecog_onset.keys()
    if block_ids != audio_onset.keys():
        raise ValueError(
            "Mismatch between ECoG and audio blocks."
        )
    all_ecog_onset = np.concatenate([ecog_onset[b] for b in block_ids], axis=0)
    all_ecog_pre = np.concatenate([ecog_pre[b] for b in block_ids], axis=0)
    all_audio_onset = np.concatenate([audio_onset[b] for b in block_ids], axis=0)
    all_audio_pre = np.concatenate([audio_pre[b] for b in block_ids], axis=0)
    all_syllable = np.concatenate([syllable_labels[b] for b in block_ids], axis=0)
    all_tone = np.concatenate([tone_labels[b] for b in block_ids], axis=0)
    min_label = np.min(all_tone)
    if min_label > 0:
        all_tone -= min_label

    output = {
        'ecog_onset': all_ecog_onset,
        'ecog_pre': all_ecog_pre,
        'ecog_sf': ecog_sf,
        'audio_onset': all_audio_onset,
        'audio_pre': all_audio_pre,
        'audio_sf': audio_sf,
        'syllable': all_syllable,
        'tone': all_tone,
    }
    if params.output_path is not None:
        np.savez(params.output_path, **output)
        print(f"Samples saved to {params.output_path}")
    return output
