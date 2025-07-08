"""
Entries required in the config file:
- `mel_kwargs` (dict): Dictionary with Mel spectrogram parameters.
- `tone_dynamic_mapping` (dict): Dictionary mapping tone labels to their dynamics.
- `n_syllables` (int): Number of syllables in the dataset.
 -`n_tones` (int): Number of tones in the dataset.
"""

import argparse
import os
from scipy.io.wavfile import write as write_wave
import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset
import json

from data_loading.dataloaders import split_dataset
from data_loading.channel_selection import select_non_discriminative_channels
from utils.utils import set_seeds
from utils.visualise import plot_training_losses
from utils.audio import mel_to_audio, compare_mels, audio_to_mel
from models.synthesisModels import SynthesisModelCNN, SynthesisLite
from models.synthesisTrainer import SynthesisTrainer
from models.simple_classifiers import ShallowNNClassifier, LogisticRegressionClassifier
from models.deep_classifiers import CNNClassifier, CNNRNNClassifier


parser = argparse.ArgumentParser(
    description="Train an audio synthesizer on ECoG data."
)
# ----- I/O -------
parser.add_argument(
    '--sample_path', type=str, required=True,
    help='Path to the .npz file containing ECoG and audio samples.'
    ' The file should contain "ecog" and "audio" keys.'
)
parser.add_argument(
    '--subject_id', type=str, required=True,
    help='ID of the subject for which the model is trained.'
)
parser.add_argument(
    '--result_file', type=str, required=True,
    help='Path to the CSV file where results will be saved. '
)
parser.add_argument(
    '--figure_dir', type=str, required=False,
    default=None,
    help='Directory to save the figures. If not provided, '
    'figures will not be saved.'
)
parser.add_argument(
    '--audio_dir', type=str, required=False,
    default=None,
    help='Directory to save the .wav audio files of '
    'synthesised waveforms. If not provided, audio files will not be saved.'
)
parser.add_argument(
    '--channel_file', type=str, default='channel_selections.json',
    help='JSON file containing channel selections for the model. '
)
parser.add_argument(
    '--config_file', type=str, default='config.json',
    help='Path to the JSON file with necessary hyperparameters.'
)
parser.add_argument(
    '--model_name', type=str, required=True,
    help='Name of the synthesis model, will be used to identify the model in the csv file.'
)
parser.add_argument(
    '--syllable_model_path', type=str, default=None,
    help='Path to the pre-trained syllable classification model. '
    'If not provided, the model will be trained from scratch.'
)
parser.add_argument(
    '--tone_model_path', type=str, default=None,
    help='Path to the pre-trained tone classification model. '
    'If not provided, the model will be trained from scratch.'
)
parser.add_argument(
    '--synthesis_model_name', type=str, required=True,
    help='Name of the synthesizer classification model to use'
    'Must match the syllable_model_path if provided.'
)
parser.add_argument(
    '--syllable_model_name', type=str, required=True,
    help='Name of the syllable classification model to use'
    'Must match the syllable_model_path if provided.'
)
parser.add_argument(
    '--tone_model_name', type=str, required=True,
    help='Name of the tone classification model to use.'
    'Must match the tone_model_path if provided.'
)
# ----- Audio Settings -------
parser.add_argument(
    '--audio_sampling_rate', type=int, default=24414,
    help='Sample rate of the audio to restore. Default is 24414 Hz.'
)
# ----- Experiment Settings -------
parser.add_argument(
    '--seed', type=int, default=42,
    help='Random seed for reproducibility. Default is 42.'
)
parser.add_argument(
    '--repeat', type=int, default=1,
    help='Number of times to repeat the training. Default is 1.'
)
parser.add_argument(
    '--verbose', type=int, default=1,
    help='Verbosity level of the training process. ' \
    '0: Only the final accuracies, 1: Basic output each run (repeat), '
    '2: Detailed output for each epoch.'
)
# ----- Training Settings -------
parser.add_argument(
    '--train_ratio', type=float, default=0.9,
    help='Ratio of the dataset to use for training. Default is 0.9.')
parser.add_argument(
    '--device', type=str, default='cuda:0',
    help='Device to use for training. Default is "cuda". ' \
    'Use "cpu" for CPU training.'
)
parser.add_argument(
    '--batch_size', type=int, default=8,
    help='Batch size for training. Default is 8.')
parser.add_argument(
    '--epochs', type=int, default=100,
    help='Number of epochs to train the model. Default is 100.')
parser.add_argument(
    '--lr', type=float, default=0.0005,
    help='Learning rate for the optimizer. Default is 0.0005.')


synthesis_models = ['SynthesisLite', 'SynthesisFull']

if __name__ == '__main__':
    params = parser.parse_args()

    # ------- Value checks  -------
    if not os.path.exists(params.sample_path):
        raise FileNotFoundError(
            f"Data file '{params.sample_path}' does not exist.")

    if 'cuda' in params.device and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please use 'cpu' as device.")

    # -------- Create directories if they do not exist --------
    if params.figure_dir and not os.path.exists(params.figure_dir):
        os.makedirs(params.figure_dir)

    if params.audio_dir and not os.path.exists(params.audio_dir):
        os.makedirs(params.audio_dir)

    result_dir = os.path.dirname(params.result_file)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # find non-discriminative channels
    with open(params.channel_file, 'r') as f:
        channel_selections = json.load(f)

    non_discriminative_channels = select_non_discriminative_channels(
        channel_selections, ['tone_discriminative', 'syllable_discriminative'])

    print('Found {} non-discriminative channels.'.format(
        len(non_discriminative_channels)))

    # load configuration files
    with open(params.config_file, 'r') as f:
        config = json.load(f)
    
    mel_kwargs = config['mel_kwargs']
    tone_dynamic_mapping = config['tone_dynamic_mapping']
    n_syllables = config['n_syllables']
    n_tones = config['n_tones']

    # load dataset, compute Mel spectrograms
    dataset = np.load(params.sample_path)

    ecog_samples = dataset['ecog']
    ecog_non = ecog_samples[:, non_discriminative_channels, :]
    ecog_syllables = ecog_samples[:, channel_selections['syllable_discriminative'], :]
    ecog_tones = ecog_samples[:, channel_selections['tone_discriminative'], :]

    audios = dataset['audio']

    mels = []
    for i, audio in enumerate(audios):
        mel = audio_to_mel(
            audio, params.audio_sampling_rate,
            mel_kwargs=mel_kwargs)
        
        mels.append(mel)
    
    mels = np.array(mels)  # (n_samples, n_mels * n_timepoints)
    
    print('Number of Mel spectrogram coefficients', mels.shape[1:])
    
    mels_dim = mels.shape[1]  # number of Mel coefficients

    seq_length = ecog_samples.shape[2]
    n_syllable_channels = ecog_syllables.shape[1]
    n_tone_channels = ecog_tones.shape[1]

    
    syllable_model_kwargs = config.get('syllable_model_kwargs', {})

    if params.syllable_model_name == 'CNN':
        syllable_model = CNNClassifier(
            input_channels= n_syllable_channels,
            input_length=seq_length,
            n_classes=n_syllables,
            **syllable_model_kwargs
        )
    elif params.syllable_model_name == 'ShallowNN':
        syllable_model = ShallowNNClassifier(
            input_dim=n_syllable_channels * seq_length,
            n_classes=n_syllables,
            **syllable_model_kwargs
        )
    elif params.syllable_model_name == 'logistic':
        syllable_model = LogisticRegressionClassifier(
            input_dim=n_syllable_channels * seq_length,
            n_classes=n_syllables,
            **syllable_model_kwargs
        )
    elif params.syllable_model_name == 'CNNRNN':
        syllable_model = CNNRNNClassifier(
            input_channels=n_syllable_channels,
            input_length=seq_length,
            n_classes=n_syllables,
            **syllable_model_kwargs
        )
    else:
        raise ValueError(
            f"Unknown syllable model name: {params.syllable_model_name}. "
            "Supported models: CNN, ShallowNN, logistic, CNNRNN."
        )
    
    tone_model_kwargs = config.get('tone_model_kwargs', {})
    if params.tone_model_name == 'CNN':
        tone_model = CNNClassifier(
            input_channels=n_tone_channels,
            input_length=seq_length,
            n_classes=n_tones,
            **tone_model_kwargs
        )
    elif params.tone_model_name == 'ShallowNN':
        tone_model = ShallowNNClassifier(
            input_dim=n_tone_channels * seq_length,
            n_classes=n_tones,
            **tone_model_kwargs
        )
    elif params.tone_model_name == 'logistic':
        tone_model = LogisticRegressionClassifier(
            input_dim=n_tone_channels * seq_length,
            n_classes=n_tones,
            **tone_model_kwargs
        )
    elif params.tone_model_name == 'CNNRNN':
        tone_model = CNNRNNClassifier(
            input_channels=n_tone_channels,
            input_length=seq_length,
            n_classes=n_tones,
            **tone_model_kwargs
        )
    else:
        raise ValueError(
            f"Unknown tone model name: {params.tone_model_name}. "
            "Supported models: CNN, ShallowNN, logistic, CNNRNN."
        )

    train_classifiers = True
    
    if params.syllable_model_path is not None:
        syllable_model.load_state_dict((torch.load(params.syllable_model_path)))
    if params.tone_model_path is not None:
        tone_model.load_state_dict((torch.load(params.tone_model_path)))

    if params.syllable_model_path is not None and params.tone_model_path is not None:
        # use pre-trained weights
        train_classifiers = False

    n_samples, n_channels, n_timepoints = ecog_non.shape

    if params.verbose > 0:
        print(
            f"Prepared {n_samples} ECoG samples with "
            f"shape {ecog_samples.shape[1:]}")

    ecog_non = torch.tensor(ecog_non, dtype=torch.float32)
    ecog_syllables = torch.tensor(ecog_syllables, dtype=torch.float32)
    ecog_tones = torch.tensor(ecog_tones, dtype=torch.float32)
    mels = torch.tensor(mels, dtype=torch.float32)

    dataset = TensorDataset(ecog_non, ecog_syllables, ecog_tones, mels)

    # repeated trainings with different splits
    mcds = []
    losses = []

    np.random.seed(params.seed)
    seeds = np.random.randint(0, 10000, params.repeat)

    for i, seed in enumerate(seeds):
        set_seeds(seed)

        ratios = [params.train_ratio, 1 - params.train_ratio]
        dataloaders = split_dataset(
            dataset, ratios, shuffling=[True, False],
            batch_size=params.batch_size,
            seed=seed
        )

        if params.synthesis_model_name == 'SynthesisLite':
            model = SynthesisLite(
                output_dim=mels_dim, n_channels=n_channels,
                n_timepoints=n_timepoints
            )
        elif params.synthesis_model_name == 'SynthesisFull':
            model = SynthesisModelCNN(
                output_dim = mels_dim, n_channels=n_channels,
                n_timepoints=n_timepoints
            )
        else:
            raise ValueError(
                f"Unknown synthesizer model name: {params.synthesizer_model_name}. "
                f"Supported models: {synthesis_models}."
            )

        trainer_verbose = params.verbose > 0 and i == 0
        trainer = SynthesisTrainer(
            synthesize_model=model,
            syllable_model=syllable_model,
            tone_model=tone_model,
            device=params.device,
            tone_dynamic_mapping=tone_dynamic_mapping,
            learning_rate=params.lr,
            verbose=trainer_verbose,
            train_classifiers=train_classifiers
        )

        if params.verbose > 0:
            print(f"Training synthesizer with seed {seed}...")

        history = trainer.train(
            dataloaders[0], params.epochs, verbose=params.verbose > 1)

        mcd, recon_mels, origin_mels = trainer.evaluate(dataloaders[1])

        mcds.append(mcd)

        if params.verbose > 0:
            print(
                f"Finished trial {i+1} / {params.repeat}. "
                f"MCD: {mcd:.4f} dB"
            )
        
        losses.append([loss for loss, _ in history])
        
    mean_mcd = np.mean(mcds)
    std_mcd = np.std(mcds)

    total_model_size = (
        model.get_nparams() + syllable_model.get_nparams() + \
        tone_model.get_nparams())
    results = {
        'model_name': params.model_name,
        'model_size': total_model_size,
        'tone_model' : params.tone_model_name,
        'tone_model_kwargs' : str(tone_model_kwargs),
        'syllable_model' : params.syllable_model_name,
        'syllable_model_kwargs' : str(syllable_model_kwargs),
        'subject': params.subject_id,
        'mel_kwargs': str(mel_kwargs),
        'seeds' : str(seeds.tolist()),
        'batch_size': params.batch_size,
        'epochs': params.epochs,
        'learning_rate': params.lr,
        'mcd_mean' : mean_mcd,
        'mcd_std' : std_mcd,
        'all_mcds': str(mcds),
    }

    results_df = pd.DataFrame([results])

    if os.path.exists(params.result_file):
        results_df.to_csv(
            params.result_file, mode='a', header=False, index=False)
    else:
        results_df.to_csv(
            params.result_file, mode='w', header=True, index=False)
    print('Saved results to ', params.result_file)

    print(f"-------- Training completed over {params.repeat} runs --------")
    print(
        f"MCD (Mel-Cepstral Distortion): {mean_mcd:.4f} dB ± {std_mcd:.4f} dB"
    )

    if params.figure_dir:
        loss_figure_path = os.path.join(params.figure_dir, 'training_losses.png')
        plot_training_losses(losses, figure_path=loss_figure_path)
        print("Saved training losses figure to ", loss_figure_path)

    n_samples = 10

    for i in range(n_samples):
        if params.audio_dir:
            origin_mel = origin_mels[i]
            recon_mel = recon_mels[i]

            origin_wave = mel_to_audio(origin_mel, mel_kwargs['n_mels'])
            recon_wave = mel_to_audio(recon_mel, mel_kwargs['n_mels'])

            audio_file_path = os.path.join(
                params.audio_dir, f'origin_audio_{i}.wav'
            )

            recon_audio_file_path = os.path.join(
                params.audio_dir, f'recon_audio_{i}.wav'
            )

            write_wave(audio_file_path, params.audio_sampling_rate, origin_wave)
            print("Saved original audio to ", audio_file_path)
            write_wave(recon_audio_file_path, params.audio_sampling_rate, recon_wave)
            print("Saved reconstructed audio to ", recon_audio_file_path)

        if params.figure_dir:
            mel_fig_path = os.path.join(
                params.figure_dir, f'mel_{i}.png'
            )

            origin_mel = origin_mel.reshape(mel_kwargs['n_mels'], -1)
            recon_mel = recon_mel.reshape(mel_kwargs['n_mels'], -1)
            
            compare_mels(
                origin_mel, recon_mel, audio_sampling_rate=params.audio_sampling_rate,
                title1="Original Mel Spectrogram",
                title2="Synthesized Mel Spectrogram",
                file_path=mel_fig_path
                )

            print("Saved original mel spectrograms to ", mel_fig_path)
