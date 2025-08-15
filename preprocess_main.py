"""Entry point for configurable preprocessing pipelines."""

import importlib
import sys
from utils.config import load_config, dict_to_namespace


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    pre_cfg = cfg.get("preprocess", {}).get("params", {})

    pipeline_cfg = pre_cfg.get("pipeline", {})
    io_cfg = pre_cfg.get("io", {})
    preprocessor_cfg = pre_cfg.get("preprocessor", {"module": "preprocess.preprocessor"})
    modalities_cfg = pre_cfg.get("modalities", {})

    pipeline_module = importlib.import_module(pipeline_cfg.get("module"))
    preprocessor_module = importlib.import_module(preprocessor_cfg.get("module"))
    io_module = importlib.import_module(io_cfg.get("module"))

    pipeline_params = dict_to_namespace(pipeline_cfg.get("params", {}))
    io_params = dict_to_namespace(io_cfg.get("params", {}))

    pipeline_module.run(
        pipeline_params, io_params, io_module, preprocessor_module, modalities_cfg
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python preprocess_main.py <config.yaml>")
    main(sys.argv[1])
