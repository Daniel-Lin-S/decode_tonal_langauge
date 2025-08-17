import os
import numpy as np
import tdt
from argparse import Namespace


def load_block(block_path: str, params: Namespace) -> dict:
    """Read a TDT block and return the data as a dictionary."""
    block_data = tdt.read_block(block_path)

    assert "streams" in dir(block_data), "No streams found in the TDT block."
    assert hasattr(block_data.streams, "EOG1"), (
        f"ECoG stream 'EOG1' not found in the TDT block {block_path}."
    )
    assert hasattr(block_data.streams, "ANIN") or hasattr(block_data.streams, "Wav1"), (
        f"Audio stream 'ANIN' or 'Wav1' not found in the TDT block {block_path}."
    )

    if hasattr(block_data.streams, "ANIN"):
        data = {
            "ecog": block_data.streams.EOG1.data,
            "audio": block_data.streams.ANIN.data[:1, :],
            "ecog_sf": block_data.streams.EOG1.fs,
            "audio_sf": block_data.streams.ANIN.fs,
        }
    elif hasattr(block_data.streams, "Wav1"):
        data = {
            "ecog": block_data.streams.EOG1.data,
            "audio": block_data.streams.Wav1.data[:1, :],
            "ecog_sf": block_data.streams.EOG1.fs,
            "audio_sf": block_data.streams.Wav1.fs,
        }

    for key, value in data.items():
        if not key.endswith("sf"):
            print(f"Shape of {key}: ", value.shape)
    return data


def save_block(
        setup_dir: str, subject_id: int,
        block_id: int, data_dict: dict,
        params: Namespace
    ) -> None:
    """Save all modalities in a block to disk."""
    subject_output_dir = os.path.join(setup_dir, f"subject_{subject_id}")
    os.makedirs(subject_output_dir, exist_ok=True)

    for key, value in data_dict.items():
        if key.endswith("_sf"):
            continue
        sf = data_dict.get(f"{key}_sf")
        file_path = os.path.join(subject_output_dir, f"B{block_id}_{key}.npz")
        np.savez(file_path, data=value, sf=sf)
        print(f"Saved {key} data to: {file_path}")
