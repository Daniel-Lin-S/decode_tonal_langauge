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


def run(config: dict) -> None:
    """Extract and preprocess ECoG signals based on configuration."""

    pre_cfg = config.get("preprocess", {})
    io_cfg = pre_cfg.get("io", {})

    params = dict_to_namespace(io_cfg)

    os.makedirs(params.output_dir, exist_ok=True)

    setup_name = generate_setup_name(pre_cfg)
    setup_dir = os.path.join(params.output_dir, setup_name)
    os.makedirs(setup_dir, exist_ok=True)

    # Save the configuration used for this setup
    config_file_path = os.path.join(setup_dir, 'config.yaml')
    with open(config_file_path, 'w') as f:
        yaml.dump(pre_cfg, f)

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
                }
            )

            print('Audio shape: ', audio.shape)
            print('ECoG data shape: ', data.shape)
            print('ECoG sampling frequency:', ecog_freq)

            for step in pre_cfg.get('steps', []):
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

                module = importlib.import_module(module_name)
                data = module.run(data, block_params)

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


if __name__ == "__main__":
    import sys
    from utils.config import load_config

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python preprocess.py <config.yaml>")
    cfg = load_config(sys.argv[1])
    run(cfg)
