"""Extract pitch contours from subject-wise samples and plot by tone labels."""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import parselmouth

from utils.config import (
    load_config,
    update_configuration,
    generate_hash_name_from_config,
)


def compute_pitch_contours(
    audio: np.ndarray,
    sf: float,
    pitch_params: Dict,
) -> Tuple[np.ndarray, float]:
    """Compute pitch contours for each audio sample."""
    contours = []
    pitch_sf = None
    expected = None
    for sample in audio:
        sound = parselmouth.Sound(sample, sampling_frequency=sf)
        pitch_obj = sound.to_pitch(
            time_step=pitch_params.get("time_step", 0.01),
            pitch_floor=pitch_params.get("pitch_floor", 75),
            pitch_ceiling=pitch_params.get("pitch_ceiling", 500),
        )
        values = pitch_obj.selected_array["frequency"]
        if pitch_sf is None:
            pitch_sf = 1.0 / pitch_obj.get_time_step()
            expected = values.shape[0]
        elif values.shape[0] != expected:
            raise ValueError(
                "Pitch contour length mismatch across samples: "
                f"expected {expected}, got {values.shape[0]}"
            )
        contours.append(values)
    return np.array(contours), pitch_sf


def plot_pitch_by_tone(
    pitch: np.ndarray,
    tones: np.ndarray,
    pitch_sf: float,
    save_path: str,
) -> None:
    """Plot mean pitch contour with s.e.m. per tone."""
    time = np.arange(pitch.shape[1]) / pitch_sf
    unique_tones = np.unique(tones)
    plt.figure(figsize=(10, 6))
    for t in unique_tones:
        tone_pitch = pitch[tones == t]
        mean = np.nanmean(tone_pitch, axis=0)
        sem = np.nanstd(tone_pitch, axis=0) / np.sqrt(tone_pitch.shape[0])
        plt.plot(time, mean, label=f"Tone {t}")
        plt.fill_between(time, mean - sem, mean + sem, alpha=0.3)
    plt.xlabel("Time (s)")
    plt.ylabel("Pitch (Hz)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def run(config: dict) -> str:
    """Run pitch extraction from configuration."""
    print("Running extract_pitch ...")
    pitch_cfg = config.get("pitch_extraction", {})
    params = pitch_cfg.get("params", {})

    io_cfg = params.get("io", {})
    sample_dir = io_cfg.get("sample_dir", "data/samples")
    output_dir = io_cfg.get("output_dir", "data/pitch")
    pitch_params = params.get("pitch", {})

    sample_cfg_path = os.path.join(sample_dir, "config.yaml")
    base_cfg = load_config(sample_cfg_path) if os.path.exists(sample_cfg_path) else {}

    hash_name = generate_hash_name_from_config(
        "ecog_pitch", {**base_cfg, "pitch": pitch_params}
    )
    out_dir = os.path.join(output_dir, hash_name)
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figure")
    os.makedirs(fig_dir, exist_ok=True)

    update_configuration(
        output_path=os.path.join(out_dir, "config.yaml"),
        previous_config_path=sample_cfg_path,
        new_module="pitch_extraction",
        new_module_cfg=pitch_cfg,
    )

    subject_files = [
        f for f in os.listdir(sample_dir)
        if f.startswith("subject_") and f.endswith(".npz")
    ]
    if not subject_files:
        raise FileNotFoundError(
            f"No subject files found in {sample_dir}."
        )

    for file in subject_files:
        data = np.load(os.path.join(sample_dir, file))
        ecog = data["ecog"]
        ecog_sf = data["ecog_sf"]
        audio = data["audio"]
        audio_sf = data["audio_sf"]
        syllable = data["syllable"]
        tone = data["tone"]

        pitch, pitch_sf = compute_pitch_contours(audio, audio_sf, pitch_params)
        pitch = pitch[:, None, :]

        np.savez(
            os.path.join(out_dir, file),
            ecog=ecog,
            ecog_sf=ecog_sf,
            syllable=syllable,
            tone=tone,
            pitch=pitch,
            pitch_sf=pitch_sf,
        )

        subject_id = file.split("_")[1].split(".")[0]
        plot_path = os.path.join(
            fig_dir, f"subject_{subject_id}_pitch_by_tone.png"
        )
        plot_pitch_by_tone(pitch.squeeze(1), tone, pitch_sf, plot_path)

    print(f"Pitch contours saved to {out_dir}", flush=True)
    return out_dir


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python extract_pitch.py <config.yaml>")
    cfg = load_config(sys.argv[1])
    run(cfg)
