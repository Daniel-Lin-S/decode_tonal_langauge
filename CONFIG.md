# Configuration Reference

This document describes the available parameters for each pipeline module.

## Dataset
- **syllable_labels**: Ordered list of syllable classes.
- **tone_labels**: Ordered list of tone classes.

## Preprocess (`preprocess.py`)
### **`params.io`**
- **`root_dir`**: Root directory containing the raw data for all subjects.  
- **`subject_dirs`**: List of directories for each subject containing their raw data.
- **`output_dir`**: Root directory to save processed data. 
- **`subject_ids`**: List of subject IDs used for naming output files. 
### **`params.steps`**
The preprocessing steps are performed **in order** as specified in the configuration file. Each step is defined by a module and its associated parameters. Some common preprocessing tools can be found in `preprocess`. 
- **preprocess.downsample**: downsample the signal to a specified frequency.
- **preprocess.frequency_filter**: extract a particular frequency band, using Butterworth filter or Hilbert envelope (if `envelope=True`). 
- **preprocess.channel_zscore**: Apply z-score normalisation to each channel.
- **preprocess.zscore_rereference**: Rereference each channel of the input data using a specified reference interval.

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

## Sample Collection (`extract_samples.py`)
### **`params.io`**
- **`textgrid_root`**: Root directory containing TextGrid files.
- **`output_dir`**: Directory to save extracted samples.
    - Each `.npz` file will be named `subject_{id}.npz`.
    - A configuration file `config.yaml` will also be saved in this directory with all settings used for pre-processing and sample collection.
- **`recording_dir`**: (optional) Only required when running this file separately. Directory containing ECoG and audio `.npz` files.
### **`params.settings`**
- **`syllable_identifiers`**: List of syllable identifies in the textgrid files.
    Example: `["i", "a"]`
- **`overwrite`**: Whether to overwrite existing files. Default: `false`
### **`subjects`**: subject specific parameters
- **`textgrid_dir`**: Subdirectory under `textgrid_root` containing TextGrid files for the subject.  
    Example: `subject_24`
- **`rest_period`**: Rest period (in seconds) for statistical tests.  
Example: `[0.5, 1.0]`
- **`sample_length`**: Length (in seconds) of each sample (epoch) to extract from the whole recording.
- **`start_offset`**: Offset (in seconds) to extract before the event. If not given, default is `0.0`.
- **`tier_list`**: List of tiers (`str`) in the textgrid files to consider for sample extraction. If not given, all tiers will be used.
- **`blocks`**: List of blocks (`int`) to process. If not given, all blocks will be extracted.

## Channel selection {`channel_selection.py`}
### **`params.io`**
- **`output_dir`**: Directory to save the results of channel selection.  
Example: `channel_selection_results/`
- **`figure_dir`**: Directory to save generated figures (optional).  
Example: `figures/`
- **`sample_dir`**: (optional) Only required when running this file separately. Directory containing `.npz` files for each subject.
### **`params.selections`**: The modules for selecting channels
All results will be saved into a single configuration file. 
- **`module`**: The Python module to be used for the selection. (under `channel_selection` module)
    Example: `channel_selection.active`
- **`selection_name`**: The name of the selection, used to identify the results.  
    Example: `active_channels`
- **`params`**: Parameters specific to the selection module, see the python scripts of each selection module for details.

## Model
- **`model`**: Python path to the classifier class.
- **`model_name`**: Short identifier for the model.
- **`model_kwargs`**: Additional keyword arguments for model initialisation.

## Training (`train_classifier.py`)
### **`params.io`**
- **`log_dir`**: Directory for TensorBoard logs and results (in a csv file)
- **`sample_dir`**: (optional) Only required when running this file separately. Path to the training samples, should have files `subject_{id}.npz`. 
- **`channel_selection_dir`**: (optional) Only required when running this file separately. JSON file containing channel selections for the model. If not specified, all channels will be used.
### Experiment Settings
- **`targets`**: A list of strings for the target variable(s) to predict in the `npz` files.
- **`separate_models`**: Train separate models for each target if `true`.
- **`seed`**: The global random seed.
- **`repeat`**: Number of repetitions with different seeds.
- **`verbose`**: Verbosity level. 0 for no message, 1 for basic messages, 2 for detailed messages for each run. Default is 1.
- **`device`**: Computation device (`cpu` or `cuda:0`). Cuda device id must be provided.
- **`subject_ids`**: Filter for subject ids. If not given, all subjects used.
### **`params.training`**
- **`train_ratio`**: Proportion of data used for training.
- **`vali_ratio`**: Proportion of data used for validation.
- **`test_ratio`**: Proportion of data used for testing.
- **`batch_size`**: Batch size for training.
- **`epochs`**: Maximum number of training epochs.
- **`lr`**: Base learning rate.
- **`patience`**: Patience for early stopping.
- **`log_every_n_steps`**: Logging frequency in training steps.

## Evaluation
- **`metrics`**: List of metrics to compute 
    - For classification: choose from `accuracy`, `f1`, `precision`, `recall`, `confusion_matrix`, `cohen_kappa` or use any function name of sklearn.metrics.
        - `f1`, `precision`, `recall` of multi-class labels are the weighted sum of each class.
- **`metric_aggregates`**: List of aggregation methods across repeated runs with different seeds.
    - For example, with `metrics = [accuracy], aggregates=[mean, std]`, the mean and standard deviation of accuracies will be recorded as `accuracy_mean` and `accuracy_std` in the output csv file, together with accuracies of each run in `accuracy_all`.
    - By default, `aggregates=[mean, std]`. 
    - Each value must be a function in numpy.
    - `confusion_matrix` is aggregated by pointwise addition for all runs, it ignores the aggregates parameter.
