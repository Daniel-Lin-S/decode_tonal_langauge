"""Run the full experiment pipeline based on a YAML configuration."""

import importlib
from typing import Dict, Any

from utils.config import load_config

STAGES = [
    "preprocess",
    "sample_collection",
    "channel_selection",
    "training",
    "evaluation",
    "visualisation",
]


def run_pipeline(config_path: str) -> None:
    """Execute pipeline stages defined in the configuration file."""
    config: Dict[str, Any] = load_config(config_path)
    stage_outputs: Dict[str, str] = {}

    for stage in STAGES:
        stage_cfg = config.get(stage)
        if not stage_cfg:
            continue

        module_name = stage_cfg.get("module")
        func_name = stage_cfg.get("function", "run")
        if module_name is None:
            continue

        print('----------- Running stage:', stage, '-----------')

        update_stage_cfg_io(stage_outputs, stage, stage_cfg)

        config[stage] = stage_cfg

        module = importlib.import_module(module_name)

        try:
            func = getattr(module, func_name)
        except AttributeError:
            raise ImportError(
                f"Module '{module_name}' does not have a function '{func_name}'"
                f"Available functions: {', '.join(dir(module))}"
            )

        result = func(config)
        
        if isinstance(result, str):
            stage_outputs[stage] = result


def update_stage_cfg_io(stage_outputs: dict, stage: str, stage_cfg: dict):
    if stage == "sample_collection":
        params_io = stage_cfg.setdefault("params", {}).setdefault("io", {})
        if "recording_dir" not in params_io and "preprocess" in stage_outputs:
            params_io["recording_dir"] = stage_outputs["preprocess"]
    elif stage == "channel_selection":
        io_cfg = stage_cfg.setdefault("params", {}).setdefault("io", {})
        if "sample_dir" not in io_cfg and "sample_collection" in stage_outputs:
            io_cfg["sample_dir"] = stage_outputs["sample_collection"]
    elif stage == "training":
        params_io = stage_cfg.setdefault("params", {}).setdefault("io", {})
        if "sample_dir" not in params_io and "sample_collection" in stage_outputs:
            params_io["sample_dir"] = stage_outputs["sample_collection"]
        if (
                "channel_selection_dir" not in params_io
                and "channel_selection" in stage_outputs
            ):
            params_io["channel_selection_dir"] = stage_outputs["channel_selection"]


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python run_pipeline.py <config.yaml>")
    run_pipeline(sys.argv[1])
