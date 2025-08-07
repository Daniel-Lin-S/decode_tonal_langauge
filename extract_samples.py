"""
Align ECoG and audio samples using TextGrid annotations.
Configuration is provided via YAML.
"""

import os
import yaml
import hashlib

from data_loading.text_align import handle_textgrids, extract_ecog_audio
from utils.config import dict_to_namespace, update_configuration


def run(config: dict) -> None:
    """Extract samples from multiple subjects based on configuration."""

    collection_cfg = config.get("sample_collection", {})
    params_config = collection_cfg.get("params", {})
    params_dict = {}
    for section in ("io", "settings"):
        params_dict.update(params_config.get(section, {}))
    params = dict_to_namespace(params_dict)

    if not hasattr(params, "overwrite"):
        params.overwrite = False

    output_dir_name = generate_output_dir_name(
        os.path.basename(params.recording_dir),
        collection_cfg
    )

    output_dir = os.path.join(params.output_dir, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    update_configuration(
        output_path=os.path.join(output_dir, "config.yaml"),
        previous_config_path=os.path.join(params.sample_dir, "config.yaml"),
        new_module='sample_collection',
        new_module_cfg=collection_cfg
    )

    for subject_id, subject_params in params_config.get("subjects", {}).items():
        subject_path = os.path.join(params.recording_dir, f"subject_{subject_id}")
        if not os.path.exists(subject_path):
            print(
                f"Recording directory {subject_path} not found. Skipping..."
            )
            continue

        subject_output_path = os.path.join(
            output_dir, f"subject_{subject_id}.npz"
        )
        if os.path.exists(subject_output_path) and not params.overwrite:
            print(f"Output file {subject_output_path} already exists. Skipping ...")
            continue

        textgrid_dir = os.path.join(params.textgrid_root, subject_params["textgrid_dir"])
        if not os.path.exists(textgrid_dir):
            print(f"TextGrid directory {textgrid_dir} not found. Skipping...")
            continue

        print(
            '------------------------ \n'
            f'Extracting all samples from {subject_path} using textgrids from {textgrid_dir}'
            '\n ------------------------'
        )

        intervals = handle_textgrids(
            textgrid_dir,
            start_offset=subject_params.get("start_offset", 0.0),
            tier_list=subject_params.get("tier_list", None),
            blocks=subject_params.get("blocks", None),
        )

        if len(intervals) == 0:
            raise ValueError(
                "No intervals found in the TextGrid files. "
                "Check the directory and file naming conventions."
                f"Target blocks: {params.blocks if params.blocks else 'all'}"
            )

        print(f"Extracted intervals from TextGrid files: {len(intervals)} blocks found.")

        extract_ecog_audio(
            intervals,
            subject_path,
            syllables=params.syllable_identifiers,
            length=subject_params["sample_length"],
            output_path=subject_output_path,
            rest_period=tuple(subject_params["rest_period"]),
        )


def generate_output_dir_name(base_name: str, collection_cfg: dict) -> str:
    """
    Generate a unique and human-readable name for the output directory
    based on the recording directory and sample extraction parameters.
    """
    hash_input = yaml.dump(collection_cfg, sort_keys=True)
    hash_part = hashlib.md5(hash_input.encode()).hexdigest()[:6]

    return f"{base_name}__{hash_part}"


if __name__ == "__main__":
    import sys
    from utils.config import load_config

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python extract_samples.py <config.yaml>")
    
    cfg = load_config(sys.argv[1])
    run(cfg)
