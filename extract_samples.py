"""Extract aligned ECoG and audio samples.

The extraction process is configurable so that datasets with different
annotation formats can plug in custom interval and sample extraction
modules. By default the pipeline uses TextGrid annotations via
``data_loading.text_align``.
"""

import os
import yaml
import hashlib
import importlib

from utils.config import dict_to_namespace, update_configuration


def run(config: dict) -> str:
    """Extract samples from multiple subjects based on configuration."""

    collection_cfg = config.get("sample_collection", {})
    params_config = collection_cfg.get("params", {})

    try:
        recording_dir = params_config['io']['recording_dir']
        output_dir = params_config['io']['output_dir']
    except:
        raise ValueError(
            "Configuration must contain 'io.recording_dir' and 'io.output_dir' "
            "to specify the directory with the recording data "
            "and the output directory for the extracted samples."
        )

    overwrite = params_config.get('io', {}).get('overwrite', False)

    module_cfg = params_config.get("modules", {})

    interval_cfg = module_cfg['interval_extractor']
    sample_cfg = module_cfg['sample_extractor']

    interval_module = importlib.import_module(interval_cfg["module"])
    interval_extractor = getattr(interval_module, interval_cfg.get("function", "get_intervals"))

    sample_module = importlib.import_module(sample_cfg["module"])
    sample_extractor = getattr(sample_module, sample_cfg.get("function", "get_samples"))

    output_dir_name = _generate_output_dir_name(
        os.path.basename(recording_dir),
        collection_cfg
    )

    output_dir = os.path.join(output_dir, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    figure_root = os.path.join(output_dir, 'figures')
    os.makedirs(figure_root, exist_ok=True)

    update_configuration(
        output_path=os.path.join(output_dir, "config.yaml"),
        previous_config_path=os.path.join(recording_dir, "config.yaml"),
        new_module='sample_collection',
        new_module_cfg=collection_cfg
    )

    for subject_id, subject_cfg in params_config.get("subjects", {}).items():
        subject_path = os.path.join(recording_dir, f"subject_{subject_id}")
        if not os.path.exists(subject_path):
            print(
                f"Recording directory {subject_path} not found. Skipping...",
                flush=True
            )
            continue

        subject_output_path = os.path.join(
            output_dir, f"subject_{subject_id}.npz"
        )
        if os.path.exists(subject_output_path) and not overwrite:
            print(
                f"Output file {subject_output_path} already exists. Skipping ...",
                flush=True
            )
            continue

        interval_subject_cfg = subject_cfg.get("interval_extractor", {})
        interval_subject_params = dict_to_namespace(
            interval_subject_cfg.get("params", {})
        )

        print(
            '------------------------ \n'
            f'Extracting all samples from {subject_path}'
            '\n ------------------------',
            flush=True
        )

        intervals = interval_extractor(interval_subject_params)

        if len(intervals) == 0:
            raise ValueError(
                "No intervals found in the annotation files. "
                "Check the directory and file naming conventions."
            )

        print(
            "Extracted intervals from annotation files: "
            f"{len(intervals)} blocks found.",
            flush=True
        )

        sample_subject_cfg = subject_cfg.get("sample_extractor", {}).copy()
        sample_subject_params = dict_to_namespace(
            sample_subject_cfg.get("params", {})
        )

        sample_subject_params.subject_id = subject_id
        sample_subject_params.data_dir = os.path.join(
            recording_dir, f'subject_{subject_id}')
        sample_subject_params.output_path = subject_output_path

        sample_extractor(
            intervals=intervals,
            params=sample_subject_params
        )

    return output_dir



def _generate_output_dir_name(base_name: str, collection_cfg: dict) -> str:
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
