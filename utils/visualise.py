import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
from typing import List, Optional
from matplotlib_venn import venn3
import json
import pandas as pd
import seaborn as sns


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
    plt.xlabel('Frequency (Hz)', fontsize=16)
    plt.ylabel('Power', fontsize=16)

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


def plot_metric(
        data: pd.DataFrame, metric: str, output_path: str = None,
        title: str = "Model Performance Comparison",
        chance_line: Optional[float] = None,
        plot_model_size: bool=True,
        model_name_map: Optional[dict] = None,
        legend_offset: float=0.2,
        models_to_plot: Optional[List[str]] = None
    ) -> None:
    """
    Generalized function to plot a metric (e.g., accuracy or F1 score) comparison
    for different models on each subject using box plots with error bars.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing columns:
        - 'model_name': Name of the model.
        - 'target': Target variable (e.g., subject or task)
        - 'subject': Subject identifier.
        - '<metric>_mean': Mean value of the metric.
        - '<metric>_std': Standard deviation of the metric. 
    metric : str
        The metric to plot (e.g., 'accuracy_mean' or 'f1_mean').
    output_path : str, optional
        Path to save the plot. If None, the plot will be displayed interactively.
    title : str, default="Model Performance Comparison"
        Title of the plot.
    chance_line : float, optional
        If provided, a horizontal line will be drawn at this value
        to indicate the chance level for the metric.
    plot_model_size : bool, default=True
        If True, the size of the points will be proportional to the model size.
        Requires 'model_size' column in the DataFrame.
    model_name_map : dict, optional
        A dictionary to map model names to more descriptive labels.
        If None, the original model names will be used.
    legend_offset : float, optional
        Use a larger value if the captions in your legend are longer.
    models_to_plot : List[str], optional
        If provided, only these models will be plotted.
        If None, all models in the DataFrame will be plotted.
    """
    if plot_model_size and 'model_size' not in data.columns:
        raise ValueError(
            "When plot_model_size is True, "
            "'model_size' must be a column in the DataFrame."
        )

    # filter models
    if models_to_plot is not None:
        data = data[data['model_name'].isin(models_to_plot)].copy()

    metric_mean = f"{metric}_mean"  # Mean column name
    metric_std = f"{metric}_std"  # Standard deviation column name

    plt.figure(figsize=(14, 8))

     # Ensure 'subject' is categorical and consistent
    data['subject'] = data['subject'].astype(str)
    subject_order = sorted(data['subject'].unique())
    subject_map = {subj: i for i, subj in enumerate(subject_order)}
    data['subject_idx'] = data['subject'].map(subject_map)

    # Add offset by model for visual separation
    unique_models = sorted(data['model_name'].unique())
    model_offsets = {
        model: i * 0.15 - 0.15 * (len(unique_models) - 1) / 2
        for i, model in enumerate(unique_models)
    }
    data['offset'] = data['model_name'].map(model_offsets)
    data['x_numeric'] = data['subject_idx'] + data['offset']

    if model_name_map:
        data['model_display'] = data['model_name'].map(model_name_map)
    else:
        data['model_display'] = data['model_name']

    if plot_model_size:
        ax = sns.scatterplot(
            data=data,
            x='x_numeric',
            y=metric_mean,
            hue='model_display',
            size='model_size',
            sizes=(50, 1000),
            palette='Set2',
            legend='brief'
        )
    else:
        ax = sns.scatterplot(
            data=data,
            x='x_numeric',
            y=metric_mean,
            hue='model_display',
            palette='Set2',
            legend='brief'
        )

    # Add error bars with offsets
    ax.errorbar(
        x=data['x_numeric'],
        y=data[metric_mean],
        yerr=data[metric_std],
        fmt='none',
        capsize=3,
        ecolor='gray',
        elinewidth=1,
        alpha=0.7
    )

    if chance_line is not None:
        plt.axhline(
            y=chance_line, color='red', linestyle='--',
            label='Chance Level', linewidth=2
        )

    plt.title(title, fontsize=18)
    plt.xlabel('Subject', fontsize=16)
    plt.ylabel(metric.capitalize(), fontsize=16)
    ax.set_xticks(list(subject_map.values()))
    ax.set_xticklabels(
        list(subject_map.keys()),
        fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(
        title='Model', loc='upper right',
        fontsize=14, bbox_to_anchor=(1+legend_offset, 1),
        title_fontsize=16)

    if output_path:
        plt.savefig(output_path, dpi=400, bbox_inches='tight')
    else:
        plt.show()
