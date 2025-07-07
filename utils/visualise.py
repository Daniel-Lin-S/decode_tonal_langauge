import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
from typing import List, Optional
from matplotlib_venn import venn3
import json


def plot_training_losses(
        losses: List[List[float]],
        vali_losses: Optional[List[List[float]]] = None,
        figure_path: Optional[str]=None
    ) -> None:
    """
    Plots the training losses over epochs for multiple trials
    and optionally saves the figure.

    Parameters
    -----------
    losses : List[List[float]]
        A list of lists where each inner list contains the
        loss values for a single trial across epochs.
    vali_losses : Optional[List[List[float]]], default=None
        A list of lists where each inner list contains the
        validation loss values for a single trial across epochs. \n
        If provided, it will be plotted alongside the training losses.
    figure_path : Optional[str], default=None
        The file path where the plot will be saved. \n
        If None, the plot will be displayed 
        interactively instead of being saved.
    """
    num_epochs = min(len(loss) for loss in losses)
    truncated_losses = [np.array(loss)[:num_epochs] for loss in losses]
    if vali_losses is not None:
        num_vali_epochs = min(len(vali_loss) for vali_loss in vali_losses)
        truncated_vali_losses = [
            np.array(vali_loss)[:num_vali_epochs] for vali_loss in vali_losses]

    mean_loss = np.mean(truncated_losses, axis=0)

    plt.figure(figsize=(10, 5))
    for i, loss_curve in enumerate(losses):
        # Adjust the color intensity
        color = plt.cm.Blues(0.3 + (i / len(losses)) * 0.7)  
        plt.plot(loss_curve, color=color, alpha=0.3) 

    # Solid thick line for mean loss
    plt.plot(mean_loss, color='blue', linewidth=2.5, label='Mean Loss')

    if vali_losses is not None:
        mean_vali_loss = np.mean(truncated_vali_losses, axis=0)
        plt.plot(
            mean_vali_loss, color='green',
            linewidth=2.5, label='Mean Validation Loss',
            linestyle='--')

        for i, vali_curve in enumerate(vali_losses):
            color = plt.cm.Greens(0.3 + (i / len(vali_losses)) * 0.7)  
            plt.plot(vali_curve, color=color, alpha=0.3, linestyle='--')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve Across All Repetitions')
    plt.legend()

    if figure_path:
        plt.savefig(figure_path, dpi=400)
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
        confusion_matrix: np.ndarray,
        add_numbers: bool=False,
        label_names: Optional[List[str]]=None,
        figure_path: Optional[str]=None
    ) -> None:
    """
    Plot the confusion matrix.

    Parameters
    ----------
    confusion_matrix : np.ndarray
        A 2D numpy array representing the confusion matrix.
        Shape should be (n_classes, n_classes).
    add_numbers : bool, default=False
        If True, display the numbers in each cell of the confusion matrix.
    label_names : Optional[List[str]], default=None
        A list of class names corresponding to the rows and
        columns of the confusion matrix.
        If None, numerical indices will be used as labels.
    figure_path : Optional[str], default=None
        The file path where the plot will be saved. If None, the plot will be displayed
        interactively instead of being saved.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=20)
    plt.colorbar()
    plt.xlabel('Predicted Label', fontsize=18)
    plt.ylabel('True Label', fontsize=18)

    if label_names is not None:
        plt.xticks(np.arange(len(label_names)), label_names, rotation=45)
        plt.yticks(np.arange(len(label_names)), label_names)
    else:
        plt.xticks(np.arange(confusion_matrix.shape[1]))
        plt.yticks(np.arange(confusion_matrix.shape[0]))

    if add_numbers:
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                plt.text(
                    j, i, f"{confusion_matrix[i, j]:.0f}",
                    ha="center", va="center",
                    color="white" if confusion_matrix[
                        i, j] > confusion_matrix.max() / 2 else "black")

    plt.grid(False)
    plt.tight_layout()

    if figure_path:
        plt.savefig(figure_path, dpi=400)
        plt.close()
    else:
        plt.show()

def plot_psd(
        data: np.ndarray,
        figure_path: Optional[str]=None
    ) -> None:
    """
    Plot the power-spectrum density given the signal data

    Parameters
    ----------
    data : np.ndarray
        Input data, shape (n_channels, n_timepoints)
    figure_path : Optional[str], default=None
        The file path where the plot will be saved. \n
        If None, the plot will be displayed
        interactively instead of being saved.
    """
    n_channels = data.shape[0]

    cmap = get_cmap('viridis', n_channels)

    plt.figure(figsize=(12, 6))
    
    for i in range(n_channels):
        f, Pxx = plt.psd(data[i], NFFT=256, Fs=1000, noverlap=128)
        # ignore the DC component (f=0)
        plt.plot(f[1:], Pxx[1:], color=cmap(i))

    plt.title('Power Spectrum Density (PSD)', fontsize=18)
    plt.xlabel('Frequency (Hz)', fontsize=15)
    plt.ylabel('Power', fontsize=15)

    if figure_path:
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_channel_mean_std(
        data: np.ndarray, sampling_rate: int,
        start: float = 0.0, end: float=30.0,
        figure_path: Optional[str] = None
    ) -> None:
    """
    Plot the change of mean and standard deviation of each channel over time.

    Parameters
    ----------
    data : np.ndarray
        ECoG data of shape (n_channels, n_timepoints).
    sampling_rate : int
        Sampling rate of the data (samples per second).
    start : float, default=0.0
        Start time in seconds for the plot.
    end : float, default=30.0
        End time in seconds for the plot.
    figure_path : Optional[str], default=None
        The file path where the plot will be saved. \n
        If None, the plot will be displayed
        interactively instead of being saved.

    Returns
    -------
    None
    """
    n_channels, _ = data.shape
    segment_length = sampling_rate  # 1-second segments
    n_segments = int(end - start)  # Number of 1-second segments

    # Initialise arrays to store mean and std for each channel and segment
    means = np.zeros((n_channels, n_segments))
    stds = np.zeros((n_channels, n_segments))

    # Compute mean and std for each channel and segment
    for i in range(n_segments):
        a = start + i * segment_length
        b = a + segment_length
        segment = data[:, a:b]
        means[:, i] = np.mean(segment, axis=1)
        stds[:, i] = np.std(segment, axis=1)

    # Create the figure
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    im1 = ax[0].imshow(means, aspect='auto', cmap='viridis', origin='lower')
    ax[0].set_title('Mean Over Time', fontsize=14)
    ax[0].set_xlabel('Time (seconds)', fontsize=12)
    ax[0].set_ylabel('Channels', fontsize=12)
    fig.colorbar(im1, ax=ax[0], orientation='vertical', label='Mean')

    im2 = ax[1].imshow(stds, aspect='auto', cmap='plasma', origin='lower')
    ax[1].set_title('Standard Deviation Over Time', fontsize=14)
    ax[1].set_xlabel('Time (seconds)', fontsize=12)
    ax[1].set_ylabel('Channels', fontsize=12)
    fig.colorbar(im2, ax=ax[1], orientation='vertical', label='Standard Deviation')

    # Adjust layout and show the plot
    plt.tight_layout()
    
    if figure_path:
        plt.savefig(figure_path, dpi=400, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_discriminative_channel(
        data : dict, channel_idx: int,
        sampling_rate: int,
        p_vals: np.ndarray,
        p_threshold: float = 0.05,
        label_name: str = 'syllable',
        recording_name: str = 'ecog',
        onset_time: Optional[int] = None,
        figure_path: Optional[str]=None
    ) -> None:
    """
    Plot the discriminative power of a specific channel over time.
    
    Parameters
    ----------
    data : dict
        Dictionary containing the recording data and labels.
        Must have keys corresponding to the recording name and label name.
    channel_idx : int
        Index of the channel to plot.
    sampling_rate : int
        Sampling rate of the recording (in Hz).
    p_vals : np.ndarray
        A one dimensional array of shape (n_timepoints, )
        with the p-values of discriminative power for the channel
        at each timepoint.
    label_name : str, optional
        Name of the label to test (default is 'syllable').
    recording_name : str, optional
        Name of the recording to test (default is 'ecog').
    onset_time : int, optional
        The time of event onset in seconds.
        (relative to the start of the recording)
    figure_path : str, optional
        Path to save the figure. If None, the figure
        will be plotted but not saved.
    """
    _, axes = plt.subplots(1, 2, figsize=(12, 6))

    # First plot: Mean and SEM for each label
    series = data[recording_name]
    labels = data[label_name]

    unique_labels = np.unique(labels)

    n_timepoints = series.shape[2]

    if onset_time is not None:
        timepoints = np.arange(n_timepoints) / sampling_rate - onset_time
        timepoints = timepoints.astype(float)
    else:
        timepoints = np.arange(n_timepoints) / sampling_rate

    for label in unique_labels:
        label_data = series[labels == label, channel_idx, :]
        mean_data = np.mean(label_data, axis=0)
        std_data = np.std(label_data, axis=0)
        sem_data = std_data / np.sqrt(label_data.shape[0])

        axes[0].plot(timepoints, mean_data, label=f'{label_name} {label}')
        axes[0].fill_between(
            timepoints, mean_data - sem_data, mean_data + sem_data,
            alpha=0.2, label=f'{label_name} {label} ±1 SEM'
        )

    if onset_time is not None:
        axes[0].axvline(x=0, color='k', linestyle='--', label='Onset')

    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()

    # Second plot: P-values
    axes[1].plot(timepoints, p_vals, label='P-values', color='r')
    axes[1].axhline(y=p_threshold, color='k', linestyle='--',
                    label='Significance Threshold')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('p-value')
    axes[1].legend()

    # set super title 
    plt.suptitle(
        f'Discriminative Power for Channel {channel_idx} '
        f'in distinguishing {label_name}', fontsize=18)

    # Save or show the figure
    if figure_path:
        plt.savefig(figure_path, dpi=500)
        plt.close()
    else:
        plt.show()


def plot_rest_erp(
        data: dict, rest_recording_name: str,
        erp_recording_name: str, 
        channel_idx: int,
        p_vals: list,
        p_val_threshold: float=0.05,
        sampling_rate: int=400,
        figure_path: Optional[str]=None
    ) -> None:
    """
    Compare the activity of rest and ERP recordings for a given channel by plotting
    the mean activity ± SEM.

    Parameters
    ----------
    data : dict
        Dictionary containing the rest recording and ERP recording data.
    rest_recording_name : str
        Name of the rest recording column (e.g., 'ecog_rest').
    erp_recording_name : str
        Name of the ERP recording column (e.g., 'ecog').
    channel_idx : int
        Index of the channel to compare.
    p_vals : list
        List of p-values for the channel over time.
    p_val_threshold : float, default=0.05
        The threshold used when determining the significance of each channel.
    sampling_rate : int
        Sampling rate of the data (default: 400 Hz).
    figure_path : Optional[str], default=None
        The file path where the plot will be saved.
        If None, the plot will be displayed
        interactively instead of being saved.

    Returns
    -------
    None
    """
    rest_data = data[rest_recording_name][:, channel_idx, :]
    erp_data = data[erp_recording_name][:, channel_idx, :]

    if rest_data.shape[1] != erp_data.shape[1]:
        raise ValueError(
            "Rest and ERP data must have the same number of timepoints.")
    
    n_timepoints = rest_data.shape[1]

    rest_mean = rest_data.mean(axis=0)
    rest_sem = rest_data.std(axis=0) / np.sqrt(rest_data.shape[0])

    erp_mean = erp_data.mean(axis=0)
    erp_sem = erp_data.std(axis=0) / np.sqrt(erp_data.shape[0])

    time = np.linspace(0, n_timepoints / sampling_rate, n_timepoints)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot Rest and ERP activity
    axes[0].plot(time, rest_mean, label=f'{rest_recording_name} (Rest)', color='blue')
    axes[0].fill_between(
        time, rest_mean - rest_sem, rest_mean + rest_sem, color='blue', alpha=0.2)

    axes[0].plot(time, erp_mean, label=f'{erp_recording_name} (ERP)', color='orange')
    axes[0].fill_between(
        time, erp_mean - erp_sem, erp_mean + erp_sem, color='orange', alpha=0.2)

    axes[0].set_title(f'Comparison of Rest and ERP Activity for Channel {channel_idx}',
                      fontsize=16)
    axes[0].set_xlabel('Time (s)', fontsize=14)
    axes[0].set_ylabel('Amplitude', fontsize=14)
    axes[0].legend()
    axes[0].grid(True)

    # Plot p-values
    axes[1].plot(time, p_vals, label='P-values', color='red')
    axes[1].axhline(
        y=p_val_threshold, color='black', linestyle='--',
        label='Significance Threshold')
    axes[1].set_title('P-values Over Time', fontsize=16)
    axes[1].set_xlabel('Time (s)', fontsize=14)
    axes[1].set_ylabel('P-value', fontsize=14)
    axes[1].legend()
    axes[1].grid(True)

    # Save or show the figure
    if figure_path:
        plt.savefig(figure_path, dpi=400, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_channel_venn_diagram(
        json_file_path: str,
        figure_path: str
    ):
    """
    Plot a Venn diagram of the channels in each key from the JSON file.
    This could be the file generated by `find_discriminative_channels.py`
    and `find_active_channels.py`.

    Parameters
    ----------
    json_file_path : str
        Path to the JSON file containing channel selections.
    figure_path : str
        Path to save the generated Venn diagram figure.

    Returns
    -------
    None
    """
    # Load data from the JSON file
    with open(json_file_path, 'r') as f:
        channel_data = json.load(f)

    # Extract sets of channels for each key
    syllable_discriminative = set(channel_data["syllable_discriminative"])
    tone_discriminative = set(channel_data["tone_discriminative"])
    active_channels = set(channel_data["active_channels"])

    # Create a Venn diagram
    plt.figure(figsize=(8, 8))
    venn = venn3(
        [syllable_discriminative, tone_discriminative, active_channels],
        ('Syllable Discriminative', 'Tone Discriminative', 'Active Channels')
    )

    for label_id in ['100', '010', '001', '110', '101', '011', '111']:
        label = venn.get_label_by_id(label_id)
        if label:  # Check if the label exists
            label.set_fontsize(16)

    for set_label in venn.set_labels:
        if set_label:  # Check if the label exists
            set_label.set_fontsize(16)

    # Customize the plot
    plt.title("Venn Diagram of Channel Selections", fontsize=18)
    
    if figure_path:
        plt.savefig(figure_path, dpi=400)
        plt.close()
    else:
        plt.show()
