# Tonal Language Decoding
Decode a tonal language from brain signals.

This is an unofficial implementation of the paper Yan Liu et al. ,Decoding and synthesizing tonal language speech from brain activity.Sci. Adv.9,eadh0478(2023).DOI:10.1126/sciadv.adh0478
https://www.science.org/doi/full/10.1126/sciadv.adh0478 

Features:
- Modular-level implementation for flexible extensions
- modules for pre-processing ECoG signals
- modules for aligning signals with event onsets to obtain the Event-related Potentials (ERPs)
- modules for selection of channels based on their activity and discriminative power on a categorical label (e.g. tone in this case)
These modules can be applied to other tasks.
- Extendable classifier, speech synthesizer frameworks with corresponding trainers to define your own model architectures.
- A small toolbox for visualisation.

## Environment Setup

It is recommended to use a virtual environment to manage dependencies. Run the following command in your terminal / shell to create a virtual environment:
```shell
python3 -m venv ecog_speech

# activate
source venv/bin/activate  # for Linux/MacOS
venv\Scripts\activate    # for windows

# install required packages
pip install -r requirements.txt
pip list   # check whether properly installed
```
Python 3.11.13 is used when writing this repository.

## The pipeline
All steps are modularised, implemented as a module in directories such as `data_loading`, `channel_selection`, `preprocess`. 
The stages to execute and their parameters are described in a YAML configuration file and executed via the pipeline runner:
```python
python main.py <config.yaml>
```

Each step in the module could also be runned separately using the corresponding python file, to avoid executing identical preprocessing steps repeatedly. You could also create separate configurations for each module.


### The configuration file
Configuration files are written in YAML and serve as the central place for defining datasets, pipeline stages and model parameters. Each top‑level key corresponds to a stage and contains a `module` field with the Python import path of the module to run. Additional parameters are provided under `params` or specialised sub‑keys. See [`CONFIG.md`](CONFIG.md) for a detailed reference of all available parameters.

Example excerpt:
```yaml
dataset:
  syllable_labels: ["mi", "ma"]
  tone_labels: ["tone1", "tone2", "tone3", "tone4"]

model:
  model: models.simple_classifiers.LogisticRegressionClassifier
  model_name: logistic
  model_kwargs: {}

training:
  module: train_classifier
  params:
    io:
      sample_path: data/samples/samples.npz
      figure_dir: figures
      result_file: results.csv
    experiment:
      targets: ["syllable"]
      seed: 42
    training:
      batch_size: 64
      epochs: 10
      lr: 0.0005
```
See `example_config.yaml` for a full specification.

### Adding new pipeline modules
- **Preprocess**: create a Python file in `preprocess/` with a `run(data, params)` function returning the processed array. Reference the module path in the `steps` list of the `preprocess` section in the configuration file.
- **Sample collection**: write a module with a `run(config)` entry point (see `extract_samples.py`) and set `sample_collection.module` to its import path in the configuration file.
- **Channel selection**: add a file under `channel_selection/` that exposes `run(data, params)` (and optional `generate_figures`) Reference the module path in the `selections` list of the `channel_selection` section in the configuration file.
- **Models**: implement new model classes under `models/` and point the `model.model` field in the configuration to the class.

## Model Training
Training is also driven by the configuration file. The `model` section specifies the class to instantiate, the `training` section defines the module responsible for optimisation along with its parameters, and `evaluation` section defines the metrics used to evaluate the model predictions / outputs. Running `main.py` with a configuration containing these sections will automatically import the model, construct the trainer and execute training. Alternatively, you could run `train_classifier.py` directly.

New trainers needs to be written as new Python modules and referenced by import path in the YAML file.

## Models
All model architectures and trainers and located in the repository `models`.
### Classification Models
- `classifier.py`: base class for any classification model.
- `simple_classifiers.py`: Simple classifiers serving as benchmarks (e.g. logistic regression, 2-layer perceptron)
- `deep_classifiers.py`: Deep networks inherited from `ClassifierModel`, classifiers for tones and syllable (phoneme) using the architecture propsed in the paper.
    - It supports arbitrary number of tones and syllables
- `classifierTrainer.py`: Contains the class used to train and evaluate a ClassifierModel
### Audio Synthesis Models
- `synthesisModels.py`: Contains a base class `SynthesisModel` for defining any model that follows this pipeline: combine labels and non-discriminative signals to produce speech.
    - `SynthesisModelCNN`: The model following the set up in the paper, with CNN layers
    - `SynthesisLite`: A lighter version of the above model achieving similar effect.
- `synthesisTrainer.py`: Contains the class used to train and evaluate a SynthesisModel

### Defining new models
New models can be defined by inheriting base models `classifier.ClassifierModel` and `synthesisModels.SynthesisModel`. 

## Visualisation
You may find some functions in `utils.visualise` helpful for generating figures
to visualise your datasets from various aspects.

## Contact
If you have any questions, you may contact me by email daniel.kansaki@outlook.com
