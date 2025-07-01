import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
from typing import List, Optional

from scipy.io import loadmat


def plot_training_losses(
        losses: List[List[float]],
        figure_path: Optional[str]=None
    ) -> None:
    """
    Plots the training losses over epochs for multiple trials
    and optionally saves the figure.

    Parameters
    -----------
    losses : List[List[float]]
        A list of lists where each inner list contains the loss values for a single trial 
        across epochs.
    figure_path : Optional[str], default=None
        The file path where the plot will be saved. If None, the plot will be displayed 
        interactively instead of being saved.
    """

    mean_loss = np.mean(losses, axis=0)

    plt.figure(figsize=(10, 5))
    for i, loss_curve in enumerate(losses):
        # Adjust the color intensity
        color = plt.cm.Blues(0.3 + (i / len(losses)) * 0.7)  
        plt.plot(loss_curve, color=color, alpha=0.3) 

    # Solid thick line for mean loss
    plt.plot(mean_loss, color='red', linewidth=2.5, label='Mean Loss')

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
        mat_file_path: str, channel_idx: int,
        sampling_rate: int,
        label_name: str = 'syllable',
        recording_name: str = 'ecog',
        onset_time: Optional[int] = None,
        figure_path: Optional[str]=None
    ) -> None:
    """
    Plot the discriminative power of a specific channel over time.
    
    Parameters
    ----------
    mat_file_path : str
        Path to the .mat file containing the recording and labels.
    channel_idx : int
        Index of the channel to plot.
    sampling_rate : int
        Sampling rate of the recording (in Hz).
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
    data = loadmat(mat_file_path)
    series = data[recording_name]
    labels = data[label_name].squeeze()

    unique_labels = np.unique(labels)

    for label in unique_labels:
        label_data = series[labels == label, channel_idx, :]
        mean_data = np.mean(label_data, axis=0)
        std_data = np.std(label_data, axis=0)
        sem_data = std_data / np.sqrt(label_data.shape[0])

        if onset_time is not None:
            timepoints = np.arange(
                mean_data.shape[0]) / sampling_rate - onset_time
            timepoints = timepoints.astype(float)
        else:
            timepoints = np.arange(mean_data.shape[0]) / sampling_rate

        plt.plot(timepoints, mean_data, label=f'{label_name} {label}')
        plt.fill_between(
            timepoints, mean_data - sem_data, mean_data + sem_data,
            alpha=0.2, label=f'{label_name} {label} ±1 SEM'
        )
    if onset_time is not None:
        plt.axvline(x=0, color='k', linestyle='--', label='Onset')
    
    plt.title(f'Channel {channel_idx} Discriminative Power')
    plt.xlabel('Timepoints')
    plt.ylabel('Amplitude')
    plt.legend()

    if figure_path:
        plt.savefig(figure_path, dpi=500)
        plt.close()
    else:
        plt.show()

