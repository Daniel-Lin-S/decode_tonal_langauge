import os
import tdt

def get_data(data_path: str) -> dict:
    """
    Read a TDT block and return the data as a dictionary.
    """

    block_data = tdt.read_block(data_path)

    data = {
        'ecog': block_data.streams.EOG1.data,
        'audio': block_data.streams.ANIN.data[:1, :],  # mono-channel audio
        'ecog_sf': block_data.streams.EOG1.fs,
        'audio_sf': block_data.streams.ANIN.fs
    }

    for key, value in data.items():
        if not key.endswith('sf'):
            print(f'Shape of {key}: ', value.shape)

    return data


def get_block_id(data_path: str) -> str:
    """
    Extract the block ID from the data path.
    """
    try:
        return int(dir.split('-')[-1].replace('B', ''))
    except ValueError:
        print(
            f"Skipping directory '{data_path}' as it does not match expected format.",
            "Expected format: 'HS<subject_id>-<block_id>'.",
        )
        return None
