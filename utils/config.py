import yaml
from argparse import Namespace
import os
import json
import hashlib


def load_config(path: str) -> dict:
    """Load a YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def dict_to_namespace(d, exclude_keys=None):
    """Recursively convert a dictionary into an argparse.Namespace."""
    if exclude_keys is None:
        exclude_keys = set()

    if isinstance(d, dict):
        return Namespace(
            **{k: dict_to_namespace(v)
               if k not in exclude_keys else v
               for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(v) for v in d]
    else:
        return d


def append_data_json(output_file: str, output_data: dict):
    """
    Append data to a JSON file, creating it if it does not exist.
    If the file exists, it merges the new data with existing data.

    Parameters
    ----------
    output_file : str
        The path to the output JSON file.
    output_data : dict
        The data to append to the JSON file.
    """
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {}
        existing_data.update(output_data)
        with open(output_file, "w") as f:
            json.dump(existing_data, f, indent=4)
        print('Appended active channels to', output_file)
    else:
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=4)
        print('Saved active channels to', output_file)


def update_configuration(
        output_path: str, previous_config_path: str,
        new_module: str, new_module_cfg: dict
    ) -> None:
    if os.path.exists(previous_config_path):
        previous_cfg = load_config(previous_config_path)
    else:
        previous_cfg = {}
        print(f"Warning: config.yaml not found in {previous_config_path}")

    previous_cfg[new_module] = new_module_cfg

    with open(output_path, "w") as f:
        yaml.dump(previous_cfg, f)


def generate_hash_name_from_config(
        base_name: str, config: dict
    ) -> str:
    """
    Generate a unique output directory name based
    on the base name and configuration.
    """
    hash_input = json.dumps(config, sort_keys=True)
    hash_part = hashlib.md5(hash_input.encode()).hexdigest()[:6]

    return f"{base_name}__{hash_part}"
