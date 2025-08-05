# Configuration Reference

This document describes the available parameters for each pipeline module.

## Preprocess (`preprocess.py`)
### I/O
- **tdt_dir**: Directory containing TDT blocks.
- **output_dir**: Directory to save processed ECoG data.
- **subject_id**: Subject identifier used to name output files.
### Settings
- **normalisation**: Normalisation method (`zscore` or `rereference`).
- **rereference_interval**: Interval in seconds for rereferencing.
- **envelope**: Whether to apply Hilbert envelope filtering.
- **freq_ranges**: List of frequency bands to extract.
- **freq_band**: Label used for the frequency band.
- **downsample_freq**: Target sampling frequency.

## Active Channel Selection (`find_active_channels.py`)
### I/O
- **recording_file_path**: Path to the `.npz` file containing ECoG data.
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
- **recording_file_path**: Path to the `.npz` file with ECoG data.
- **channel_locations_file**: File containing electrode locations.
- **channel_locations_key**: Key in the location file for electrode positions.
- **figure_dir**: Directory to save figures.
- **output_file**: JSON file to save discriminative channels.
- **channel_output_file**: CSV file with discriminative scores for all channels.
### Settings
- **label_names**: Labels used to evaluate discriminative power.
- **recording_name**: Key for ECoG data in the `.npz` file.
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
- **foundation_weights_path**: Path to optional pretrained weights.
- **classifier_kwargs**: Additional keyword arguments for model initialisation.

## Training (`train_classifier.py`)
### I/O
- **sample_path**: Path to the samples `.npz` file.
- **figure_dir**: Directory to store plots.
- **channel_file**: Optional channel selection file.
- **result_file**: CSV file to log evaluation results.
- **model_dir**: Directory to save trained model weights.
- **log_dir**: Directory for TensorBoard logs.
### Experiment Settings
- **subject_id**: Subject identifier.
- **targets**: Target labels to predict.
- **separate_models**: Train separate models for each target if `true`.
- **seed**: Random seed.
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
- **metrics**: List of metrics to compute (e.g. `accuracy`, `f1_score`, `precision_score`, `confusion_matrix`, ...).
- **aggregates**: List of aggregation methods across runs (e.g. `mean`, `std`).

