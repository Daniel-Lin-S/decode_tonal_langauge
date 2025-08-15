import os
import importlib
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt



def preprocess_modalities(data_dict, modalities_cfg, base_params, figure_dir=None):
    """Preprocess each modality according to its type and configured steps."""
    for modality, cfg in modalities_cfg.items():
        mod_type = cfg.get("type")
        if mod_type is None:
            raise KeyError(f"Modality '{modality}' missing 'type' field in config")
        if mod_type != "signal":
            raise NotImplementedError(
                f"Modality type '{mod_type}' not supported; currently only 'signal'"
            )

        steps = cfg.get("preprocessing", {}).get("steps", [])
        if not steps:
            continue

        params = deepcopy(base_params)
        params.signal_freq = data_dict.get(f"{modality}_sf")
        mod_fig_dir = os.path.join(figure_dir, modality) if figure_dir else None

        processed, freq = preprocess_signal(
            data_dict[modality], steps, params, figure_dir=mod_fig_dir
        )
        data_dict[modality] = processed
        if freq is not None:
            data_dict[f"{modality}_sf"] = freq

    return data_dict


# TODO - this function does not visualise steps where number of channels change.
def preprocess_signal(data, steps, block_params, figure_dir=None,
                      num_channels=5, duration=1.0):
    """Apply preprocessing steps sequentially to the data."""
    for i, step in enumerate(steps):
        module_name = step['module']
        step_params = step.get('params', {})

        for key, value in step_params.items():
            if hasattr(block_params, key):
                raise ValueError(
                    f"Parameter '{key}' already exists in params. "
                    "Please ensure no conflicting parameter names "
                    "in each preprocessing step."
                )
            setattr(block_params, key, value)

        before_data = data.copy()
        before_freq = block_params.signal_freq

        module = importlib.import_module(module_name)
        data = module.run(data, block_params)

        if figure_dir and data.ndim == 2:
            visualise_preprocessing(
                before_data, before_freq, data,
                block_params, figure_dir,
                i, module_name,
                num_channels=num_channels,
                duration=duration
            )

    return data, block_params.signal_freq


def visualise_preprocessing(
    before_data: np.ndarray,
    before_freq: float,
    after_data: np.ndarray,
    block_params: object,
    block_figure_dir: str,
    step_index: int,
    module_name: str,
    num_channels: int,
    duration: float,
) -> None:
    """Visualize the effect of preprocessing on multiple channels."""
    after_freq = block_params.signal_freq

    max_time = min(
        before_data.shape[1] / before_freq,
        after_data.shape[1] / after_freq,
    )
    duration = min(duration, max_time)
    start_time = np.random.uniform(0, max_time - duration)
    end_time = start_time + duration

    fig, ax = plt.subplots(num_channels, 1, figsize=(10, 4 * num_channels), sharex=True)
    if num_channels == 1:
        ax = [ax]  # Ensure ax is iterable for a single channel

    for i in range(num_channels):
        ch_idx = np.random.randint(0, before_data.shape[0])
        before_slice = before_data[
            ch_idx,
            int(start_time * before_freq):int(end_time * before_freq),
        ]
        after_slice = after_data[
            ch_idx,
            int(start_time * after_freq):int(end_time * after_freq),
        ]
        time_before = np.linspace(
            start_time, end_time, before_slice.shape[0], endpoint=False
        )
        time_after = np.linspace(
            start_time, end_time, after_slice.shape[0], endpoint=False
        )

        ax[i].plot(time_before, before_slice, label="before", alpha=0.7)
        ax[i].plot(time_after, after_slice, label="after", alpha=0.7)
        ax[i].set_title(f"Channel {ch_idx}", fontsize=18)
        ax[i].set_ylabel("Amplitude", fontsize=14)
        ax[i].legend(fontsize=12)

    ax[-1].set_xlabel("Time (s)", fontsize=14)
    fig.suptitle(
        f"{module_name.split('.')[-1]} - Preprocessing Step {step_index + 1}",
        fontsize=20,
    )

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

    fig_path = os.path.join(
        block_figure_dir,
        f"step{step_index + 1}_{module_name.split('.')[-1]}.png",
    )
    fig.savefig(fig_path, dpi=500)
    plt.close(fig)
