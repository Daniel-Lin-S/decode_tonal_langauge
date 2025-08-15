import os
import numpy as np
import tdt


def load_block(block_path: str) -> dict:
    """Read a TDT block and return the data as a dictionary."""
    block_data = tdt.read_block(block_path)
    data = {
        "ecog": block_data.streams.EOG1.data,
        "audio": block_data.streams.ANIN.data[:1, :],
        "ecog_sf": block_data.streams.EOG1.fs,
        "audio_sf": block_data.streams.ANIN.fs,
    }
    for key, value in data.items():
        if not key.endswith("sf"):
            print(f"Shape of {key}: ", value.shape)
    return data


def save_block(
        setup_dir: str, subject_id: int,
        block_id: int, data_dict: dict
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
