"""
Align ECoG and audio samples using TextGrid annotations.
Configuration is provided via YAML.
"""

import os

from data_loading.text_align import handle_textgrids, extract_ecog_audio
from utils.config import dict_to_namespace


def run(config: dict, config_path: str | None = None) -> None:
    """Extract samples based on configuration."""

    samp_cfg = config.get("sample_collection", {}).get("params", {})
    params_dict = {}
    for section in ("io", "experiment", "settings", "training"):
        params_dict.update(samp_cfg.get(section, {}))
    params = dict_to_namespace(params_dict)

    rest_period = tuple(params.rest_period)
    syllable_identifiers = params.syllable_identifiers

    if os.path.exists(params.output_path) and not params.overwrite:
        print(f"Output file {params.output_path} already exists. Skipping ...")
        return

    print(
        '----------- '
        f'Extracting all samples from {params.textgrid_dir} and {params.recording_dir}'
        ' -----------'
    )

    output_dir = os.path.dirname(params.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    intervals = handle_textgrids(
        params.textgrid_dir,
        start_offset=0.2,
        tier_list=['success'],
        blocks=params.blocks,
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
        params.recording_dir,
        syllable_identifiers,
        audio_kwords=params.audio_kwords,
        ecog_kwords=params.ecog_kwords,
        output_path=params.output_path,
        rest_period=rest_period,
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python extract_samples.py <config.yaml>")
    from utils.config import load_config
    cfg = load_config(sys.argv[1])
    run(cfg, sys.argv[1])
