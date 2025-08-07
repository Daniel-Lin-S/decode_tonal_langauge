"""
Find the channels with responses at events compared to rest period.
Parameters are provided via a YAML configuration.
"""

import os
import numpy as np
import json

from importlib import import_module
import warnings

from utils.config import (
    dict_to_namespace, load_config,
    update_configuration, generate_hash_name_from_config
)


def run(config: dict) -> None:
    """Identify active channels using configuration settings."""

    ch_cfg = config.get("channel_selection", {})
    io_dict = ch_cfg.get("io", {})

    params = dict_to_namespace(io_dict)

    output_dir_name = generate_hash_name_from_config(
        os.path.basename(params.sample_dir), ch_cfg
    )
    output_dir = os.path.join(params.output_dir, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    figure_root = os.path.join(output_dir, "figures")
    os.makedirs(figure_root, exist_ok=True)

    update_configuration(
        output_path = os.path.join(output_dir, "config.yaml"),
        previous_config_path = os.path.join(params.sample_dir, "config.yaml"),
        new_module='channel_selection',
        new_module_cfg=ch_cfg
    )

    for file_name in os.listdir(params.sample_dir):
        if not file_name.endswith(".npz") or not file_name.startswith("subject_"):
            continue

        subject_id = file_name.split("_")[1].split(".")[0]
        sample_file_path = os.path.join(params.sample_dir, file_name)
        data = np.load(sample_file_path)

        subject_results = {}

        for module_cfg in ch_cfg.get("selections", []):
            module_name = module_cfg["module"]
            selection_name = module_cfg["selection_name"]
            module_params = module_cfg.get("params", {})

            print(
                f'Running {module_name} for subject {subject_id} '
                f'from file {sample_file_path} '
            )

            module = import_module(module_name)
            module_results = module.run(data, module_params)

            subject_results[selection_name] = module_results["selected_channels"]

            if len(subject_results[selection_name]) == 0:
                warnings.warn(
                    'No active channels found for selection '
                    f'{selection_name} in subject {subject_id}.'
                )

            module_figure_dir = os.path.join(
                figure_root, selection_name, f'subject_{subject_id}'
            )
            os.makedirs(module_figure_dir, exist_ok=True)

            if hasattr(module, 'generate_figures'):
                module.generate_figures(
                    data, module_results, module_params,
                    figure_dir=module_figure_dir
                )

        output_file = os.path.join(output_dir, f'subject_{subject_id}.json')
        with open(output_file, "w") as f:
            json.dump(subject_results, f, indent=4)

        print(f'Saved results for subject {subject_id} to {output_file}.')


if __name__ == "__main__":
    import sys
    from utils.config import load_config

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python find_active_channels.py <config.yaml>")

    cfg = load_config(sys.argv[1])
    run(cfg)
