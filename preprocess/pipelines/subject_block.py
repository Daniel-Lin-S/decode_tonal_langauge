import os
import yaml
import hashlib
from typing import Dict, Any

from utils.config import dict_to_namespace


def get_block_id(dirname: str) -> int:
    """Extract the block ID from a directory name."""
    try:
        return int(dirname.split('-')[-1].replace('B', ''))
    except ValueError:
        print(
            f"Skipping directory '{dirname}' as it does not match expected format.",
            "Expected format: 'HS<subject_id>-<block_id>'.",
        )
        return None


def iter_blocks(root_dir: str, subject_dirs, subject_ids=None):
    """Yield (subject_id, block_id, block_path) tuples."""
    if subject_ids is None:
        subject_ids = [i + 1 for i in range(len(subject_dirs))]

    for subject_id, subject_dir in zip(subject_ids, subject_dirs):
        subject_path = os.path.join(root_dir, subject_dir)
        for dir_name in os.listdir(subject_path):
            block_id = get_block_id(dir_name)
            if block_id is None:
                continue
            block_path = os.path.join(subject_path, dir_name)
            yield subject_id, block_id, block_path


def generate_setup_name(modalities_cfg: Dict[str, Any]) -> str:
    steps = []
    for mod_cfg in modalities_cfg.values():
        steps.extend(mod_cfg.get("preprocessing", {}).get("steps", []))
    readable_parts = [step["module"].split(".")[-1] for step in steps]
    readable_name = "__".join(readable_parts) if readable_parts else "raw"
    setup_str = "_".join([f"{step['module']}_{step.get('params', {})}" for step in steps])
    hash_part = hashlib.md5(setup_str.encode()).hexdigest()[:6]
    return f"{readable_name}_{hash_part}" if readable_parts else readable_name


def run(pipeline_params, io_params, io_module, preprocessor_module, modalities_cfg):
    setup_name = generate_setup_name(modalities_cfg)
    setup_dir = os.path.join(io_params.output_dir, setup_name)
    os.makedirs(setup_dir, exist_ok=True)

    figure_root = os.path.join(setup_dir, "figures")
    os.makedirs(figure_root, exist_ok=True)

    config_path = os.path.join(setup_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(
            {
                "pipeline": vars(pipeline_params),
                "io": vars(io_params),
                "modalities": modalities_cfg,
            },
            f,
        )

    for subject_id, block_id, block_path in iter_blocks(
        io_params.root_dir,
        pipeline_params.subject_dirs,
        getattr(pipeline_params, "subject_ids", None),
    ):
        print(f"Processing block {block_id} of subject {subject_id}...")

        data_dict = io_module.load_block(block_path)

        block_params = dict_to_namespace(
            {
                **vars(io_params),
                "block_id": block_id,
                "subject_id": subject_id,
            },
            exclude_keys=["root_dir", "output_dir"],
        )

        block_figure_dir = os.path.join(
            figure_root, f"subject_{subject_id}", f"block_{block_id}"
        )
        os.makedirs(block_figure_dir, exist_ok=True)

        preprocessor_module.preprocess_modalities(
            data_dict, modalities_cfg, block_params, figure_dir=block_figure_dir
        )

        io_module.save_block(setup_dir, subject_id, block_id, data_dict)

    return setup_dir
