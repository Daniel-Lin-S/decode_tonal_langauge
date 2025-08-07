"""Run the full experiment pipeline based on a YAML configuration."""

import importlib
from typing import Dict

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
    config: Dict = load_config(config_path)

    for stage in STAGES:
        stage_cfg = config.get(stage)
        if not stage_cfg:
            continue

        module_name = stage_cfg.get("module")
        func_name = stage_cfg.get("function", "run")
        if module_name is None:
            continue

        module = importlib.import_module(module_name)
        func = getattr(module, func_name)

        try:
            func(config, config_path=config_path)
        except TypeError:
            func(config)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python run_pipeline.py <config.yaml>")
    run_pipeline(sys.argv[1])
