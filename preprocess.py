"""
Extract ECoG signal from TDT blocks, preprocess, and save to .npz files.

This script now reads its parameters from a YAML configuration file.

Structure of the output directory:
processed/
├── setup_1/
|   ├── config.yaml
│   ├── subject_1/
│   │   ├── B1_ecog.npz
│   │   ├── B1_sound.npz
│   │   ├── ...
│   ├── subject_2/
│   │   ├── B1_ecog.npz
│   │   ├── B1_sound.npz
│   │   ├── ...
├── setup_2/
|   ├── config.yaml
│   ├── subject_1/
│   │   ├── B1_ecog.npz
│   │   ├── B1_sound.npz
│   │   ├── ...
│   ├── subject_2/
│   │   ├── B1_ecog.npz
│   │   ├── B1_sound.npz
│   │   ├── ...
where B stands for block / trial.
Set up name will be generated based on the preprocessing steps:
# step1__step2__step3_<hash>
"""

import os
import tdt
import numpy as np
import yaml
import importlib
from utils.config import dict_to_namespace

import hashlib
import matplotlib.pyplot as plt


def run(config: dict) -> None:
    """Extract and preprocess ECoG signals based on configuration."""

    pre_cfg = config.get("preprocess", {})
    io_cfg = pre_cfg.get("io", {})

    params = dict_to_namespace(io_cfg)

    os.makedirs(params.output_dir, exist_ok=True)

    setup_name = generate_setup_name(pre_cfg)
    setup_dir = os.path.join(params.output_dir, setup_name)
    os.makedirs(setup_dir, exist_ok=True)

    figure_root = os.path.join(setup_dir, 'figures')
    os.makedirs(figure_root, exist_ok=True)

    # Save the configuration used for this setup
    config_file_path = os.path.join(setup_dir, 'config.yaml')
    pre_configs = {'preprocess': pre_cfg}
    with open(config_file_path, 'w') as f:
        yaml.dump(pre_configs, f)

    if not hasattr(params, 'subject_ids'):
        params.subject_ids = [
            i+1 for i in range(len(params.subject_dirs))
        ]

    for subject_id, subject_dir in zip(params.subject_ids, params.subject_dirs):
        subject_dir = os.path.join(params.root_dir, subject_dir)

        for dir in os.listdir(subject_dir):
            try:
                block_id = int(dir.split('-')[-1].replace('B', ''))
                print(f'Processing block {block_id} of subject {subject_id}...')
            except ValueError:
                print(
                    f"Skipping directory '{dir}' as it does not match expected format.",
                    "Expected format: 'HS<subject_id>-<block_id>'.",
                )
                continue

            ecog_file_name = f'B{block_id}_ecog.npz'
            audio_file_name = f'B{block_id}_sound.npz'

            subject_output_dir = os.path.join(setup_dir, f"subject_{subject_id}")
            os.makedirs(subject_output_dir, exist_ok=True)

            ecog_file = os.path.join(subject_output_dir, ecog_file_name)
            audio_file = os.path.join(subject_output_dir, audio_file_name)

            block_figure_dir = os.path.join(
                figure_root, f'subject_{subject_id}', f'block_{block_id}'
            )
            os.makedirs(block_figure_dir, exist_ok=True)

            if os.path.exists(ecog_file) and os.path.exists(audio_file):
                print(f'Skipping block {block_id}, already processed.')
                continue

            block_path = os.path.join(subject_dir, dir)
            block_data = tdt.read_block(block_path)

            data = block_data.streams.EOG1.data
            ecog_freq = block_data.streams.EOG1.fs
            audio = block_data.streams.ANIN.data
            audio = audio[:1, :]   # mono-channel audio
            audio_freq = block_data.streams.ANIN.fs

            block_params = dict_to_namespace(
                {
                    **vars(params),
                    'block_id': block_id,
                    'subject_id': subject_id,
                    'signal_freq': ecog_freq
                },
                exclude_keys=['bands']
            )

            print('Audio shape: ', audio.shape)
            print('ECoG data shape: ', data.shape)
            print('ECoG sampling frequency:', ecog_freq)

            for i, step in enumerate(pre_cfg.get('steps', [])):
                module_name = step['module']
                step_params = step.get('params', {})

                for key, value in step_params.items():
                    if hasattr(block_params, key):
                        raise ValueError(
                            f"Parameter '{key}' already exists in params. "
                            "Please ensure no conflicting parameter names"
                            " in each preprocessing step."
                        )

                    setattr(block_params, key, value)

                before_data = data.copy()
                before_freq = block_params.signal_freq

                module = importlib.import_module(module_name)
                data = module.run(data, block_params)

                if data.ndim == 2:
                    visualise_preprocessing(
                        before_data, before_freq, data,
                        block_params, block_figure_dir,
                        i, module_name,
                        num_channels=5,
                        duration=1.0
                    )

            if not os.path.exists(ecog_file):
                np.savez(ecog_file, data=data, sf=block_params.signal_freq)
                print('Saved ECoG data to:', ecog_file)
            else:
                print('ECoG data already exists:', ecog_file)

            if not os.path.exists(audio_file):
                np.savez(audio_file, data=audio, sf=int(audio_freq))
                print('Saved audio data to:', audio_file)
            else:
                print('Audio data already exists:', audio_file)


def generate_setup_name(pre_cfg: dict) -> str:
    """Generate a unique name for the preprocessing setup based on the configuration."""
    steps = pre_cfg.get("steps", [])

    readable_parts = [
        step["module"].split(".")[-1] for step in steps
    ]
    readable_name = "__".join(readable_parts)

    setup_str = "_".join([f"{step['module']}_{step['params']}" for step in steps])
    hash_part = hashlib.md5(setup_str.encode()).hexdigest()[:6]

    # ensure uniqueness and avoid overly long names
    return f"{readable_name}_{hash_part}"


# TODO - this function does not visualise steps where number of channels change.
def visualise_preprocessing(
    before_data: np.ndarray,
    before_freq: float,
    after_data: np.ndarray,
    block_params: object,
    block_figure_dir: str,
    step_index: int,
    module_name: str,
    num_channels: int,
    duration: float
) -> None:
    """
    Visualize the effect of preprocessing on multiple channels.

    Args:
        before_data (np.ndarray): Data before preprocessing (channels x timepoints).
        after_data (np.ndarray): Data after preprocessing (channels x timepoints).
        block_params (object): Block parameters containing sampling frequency.
        block_figure_dir (str): Directory to save the figure.
        step_index (int): Index of the preprocessing step.
        module_name (str): Name of the preprocessing module.
        num_channels (int): Number of random channels to plot.
        duration (float): Duration of the signal to plot (in seconds).
    """
    after_freq = block_params.signal_freq

    max_time = min(
        before_data.shape[1] / before_freq,
        after_data.shape[1] / after_freq
    )
    duration = min(duration, max_time)
    start_time = np.random.uniform(0, max_time - duration)
    end_time = start_time + duration

    fig, ax = plt.subplots(num_channels, 1, figsize=(10, 4 * num_channels), sharex=True)
    if num_channels == 1:
        ax = [ax]  # Ensure ax is iterable for a single channel

    for i in range(num_channels):
        ch_idx = np.random.randint(0, before_data.shape[0])
        before_slice = before_data[
            ch_idx,
            int(start_time * before_freq):int(end_time * before_freq)
        ]
        after_slice = after_data[
            ch_idx,
            int(start_time * after_freq):int(end_time * after_freq)
        ]
        time_before = np.linspace(
            start_time, end_time, before_slice.shape[0], endpoint=False
        )
        time_after = np.linspace(
            start_time, end_time, after_slice.shape[0], endpoint=False
        )

        ax[i].plot(time_before, before_slice, label='before', alpha=0.7)
        ax[i].plot(time_after, after_slice, label='after', alpha=0.7)
        ax[i].set_title(f'Channel {ch_idx}', fontsize=18)
        ax[i].set_ylabel('Amplitude', fontsize=14)
        ax[i].legend(fontsize=12)

    ax[-1].set_xlabel('Time (s)', fontsize=14)
    fig.suptitle(
        f'{module_name.split(".")[-1]} - Preprocessing Step {step_index + 1}',
        fontsize=20
    )

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

    fig_path = os.path.join(
        block_figure_dir,
        f'step{step_index + 1}_{module_name.split(".")[-1]}.png'
    )
    fig.savefig(fig_path, dpi=500)
    plt.close(fig)



if __name__ == "__main__":
    import sys
    from utils.config import load_config

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python preprocess.py <config.yaml>")
    cfg = load_config(sys.argv[1])
    run(cfg)
