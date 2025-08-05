# Configuration Reference

This document describes the available parameters for each pipeline module.

## Preprocess (`preprocess.py`)
### I/O
- **root_dir**: Root directory containing the raw data for all subjects.  
- **subject_dirs**: List of directories for each subject containing their raw data.
- **output_dir**: Root directory to save processed data. 
- **subject_ids**: List of subject IDs used for naming output files. 
### steps
The preprocessing steps are performed **in order** as specified in the configuration file. Each step is defined by a module and its associated parameters. Some common preprocessing tools can be found in `preprocess`. 
- **preprocess.downsample**: downsample the signal to a specified frequency.
- **preprocess.frequency_filter**: extract a particular frequency band, using Butterworth filter or Hilbert envelope (if `envelope=True`). 
- **preprocess.channel_zscore**: Apply z-score normalisation to each channel.
- **preprocess.rereference**: Rereference each channel of the input data using a specified reference interval.

See detailed descriptions of parameters for each module in the corresponding python file.

To define a new preprocessing module, please create a python file under `preprocess` module and define the main function `run`.
```python
import numpy as np
from argparse import Namespace

def run(data: np.ndarray, params: Namespace) -> np.ndarray:
    # shape of data: (n_channels, n_timepoints)
    pass
```

Note that the signal frequency is available in `params.signal_freq` as a global parameter. If the sampling rate of the signal is changed by this pre-processing step, please modify `params.signal_freq` accordingly.


## Active Channel Selection (`find_active_channels.py`)
### I/O
- **recording_file_path**: Path to the `.npz` file containing the recording.
- **figure_dir**: Directory to save diagnostic figures.
- **output_file**: JSON file to save active channel indices.
### Settings
- **rest_recording_name**: Key for rest-period data in the `.npz` file.
- **erp_recording_name**: Key for event-related potential data in the `.npz` file.
- **p_threshold**: P-value threshold for significance.
- **consecutive_length_threshold**: Minimum consecutive significant length.
- **sampling_rate**: Sampling rate of the recording for plotting.

## Discriminative Channel Selection (`find_discriminative_channels.py`)
### I/O
- **recording_file_path**: Path to the `.npz` file with the recording.
- **channel_locations_file**: File containing electrode locations.
- **channel_locations_key**: Key in the location file for electrode positions.
- **figure_dir**: Directory to save figures.
- **output_file**: JSON file to save discriminative channels.
- **channel_output_file**: CSV file with discriminative scores for all channels.
### Settings
- **label_names**: Labels used to evaluate discriminative power.
- **recording_name**: Key for recording data in the `.npz` file.
- **p_thresholds**: P-value thresholds for each label.
- **consecutive_length_thresholds**: Minimum lengths of consecutive significant time points.
- **sampling_rate**: Sampling rate of the recording.
- **onset_time**: Onset time for aligning visualisations.
- **individual_figures**: Whether to save figures for each channel individually.

## Sample Collection (`extract_samples.py`)
### I/O
- **textgrid_dir**: Directory containing `TextGrid` annotation files.
- **recording_dir**: Directory containing ECoG and audio `.npz` files.
- **output_path**: Destination path for the extracted samples `.npz` file.
### Settings
- **audio_kwords**: Keywords to select audio files.
- **ecog_kwords**: Keywords to select ECoG files.
- **blocks**: List of block numbers to process.
- **overwrite**: Overwrite existing output file if `true`.
- **rest_period**: Rest period used for referencing.
- **syllable_identifiers**: Syllables to extract from annotations.
-
## Dataset
- **syllable_labels**: Ordered list of syllable classes.
- **tone_labels**: Ordered list of tone classes.

## Model
- **model**: Python path to the classifier class.
- **model_name**: Short identifier for the model.
- **model_kwargs**: Additional keyword arguments for model initialisation.

## Training (`train_classifier.py`)
### I/O
- **sample_path**: Path to the `.npz` file containing the samples.
- **figure_dir**: Directory to store plots.
- **channel_file**: JSON file containing channel selections for the model. If not specified, all channels will be used.
- **result_file**: CSV file to log evaluation results.
- **model_dir**: Directory to save trained model weights.
- **log_dir**: Directory for TensorBoard logs.
### Experiment Settings
- **subject_id**: Subject identifier.
- **targets**: Target labels to predict.
- **separate_models**: Train separate models for each target if `true`.
- **seed**: The global random seed.
- **repeat**: Number of repetitions with different seeds.
- **verbose**: Verbosity level.
- **device**: Computation device (`cpu` or `cuda:0`).
### Training Settings
- **train_ratio**: Proportion of data used for training.
- **vali_ratio**: Proportion of data used for validation.
- **test_ratio**: Proportion of data used for testing.
- **batch_size**: Batch size for training.
- **epochs**: Maximum number of training epochs.
- **lr**: Learning rate.
- **patience**: Patience for early stopping.
- **log_every_n_steps**: Logging frequency in training steps.

## Evaluation
- **metrics**: List of metrics to compute 
    - For classification: choose from `accuracy`, `f1`, `precision`, `recall`, `confusion_matrix`, `cohen_kappa` or use any function name of sklearn.metrics.
        - `f1`, `precision`, `recall` of multi-class labels are the weighted sum of each class.
- **aggregates**: List of aggregation methods across repeated runs with different seeds.
    - For example, with `metrics = [accuracy], aggregates=[mean, std]`, the mean and standard deviation of accuracies will be recorded as `accuracy_mean` and `accuracy_std` in the output csv file, together with accuracies of each run in `accuracy_all`.
    - By default, `aggregates=[mean, std]`. 
    - Each value must be a function in numpy.
    - `confusion_matrix` is aggregated by pointwise addition for all runs, it ignores the aggregates parameter.
