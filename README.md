# Tonal Language 
Decode a tonal language from brain signals.

This is an unofficial implementation of the paper Yan Liu et al. ,Decoding and synthesizing tonal language speech from brain activity.Sci. Adv.9,eadh0478(2023).DOI:10.1126/sciadv.adh0478
https://www.science.org/doi/full/10.1126/sciadv.adh0478 

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

## Data loading and Preprocessing
Pipeline of work: 
1. `preprocess.py`: extracts ECoG and audio from TDT repositories, preprocess the ECoG using Hilbert filter and re-referencing, and save them into `npz` files
2. `extract_samples.py`: Extracts event onsets and offsets from TextGrid files and build event-related potentials (ECoG) with shape `(n_samples, n_channels, n_timepoints)` and lables of shape `(n_samples,)`. One sample correspond to one event (trial). All these are stored in one `npz` file for convenience.
    - if rest period is specified, resting ECoG samples will be extracted as well for the channel selection stage.
3. `find_active_channels.py`: By tuning the p-value threshold and length threshold, selects the appropriate channels that have significantly different response in events compared to rest period. (one-way ANOVA test used)
4. `find_discriminative_channels.py`: By tuning the p-value threshold and length threshold, selects the appropriate channels that have significantly different response to different syllables and tones. (one-way ANOVA test used)

The actual functions for performing these steps are stored in `data_loading`.

## Model Training
The main scripts for model training are provided in `train_classifier.py` and `train_synthesizer.py`.
By changing the arguments passed to these files, different training settings can be used.

Example Usage:
```shell
python3 train_classifier.py \
    --sample_path data/samples.npz \
    --figure_dir figures \
    --model_dir checkpoints \
    --channel_file channel_selections.json \
    --config_file configs/general_configs.json \
    --target tones \
    --batch_size 32 \
    --lr 0.001 \
    --verbose 2 \
    --epochs 25 \
```
Detailed description of each argument can be found in the python files.

## Models
All model architectures and trainers and located in the repository `models`.
### Classification Models
- `classifier.py`: base class for any classification model.
- `toneModel.py`, `syllableModel.py`: inherited from `ClassifierModel`, classifiers tones and syllable (phoneme) using the architecture propsed in the paper.
    - It supports arbitrary number of tones and syllables
- `classifierTrainer.py`: Contains the class used to train and evaluate a ClassifierModel
### Audio Synthesis Models
- `synthesisModels.py`: Contains a base class `SynthesisModel` for defining any model that follows this pipeline: combine labels and non-discriminative signals to produce speech.
    - `SynthesisModelCNN`: The model following the set up in the paper, with CNN layers
    - `SynthesisLite`: A lighter version of the above model achieving similar effect.
- `synthesisTrainer.py`: Contains the class used to train and evaluate a SynthesisModel
### Defining new models
New models can be defined by inheriting base models `classifier.ClassifierModel` and `synthesisModels.SynthesisModel`. 

## Contact
If you have any questions, you may contact me by email daniel.kansaki@outlook.com
