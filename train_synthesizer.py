import argparse
import os
from scipy.io import loadmat
from scipy.io.wavfile import write as write_wave
import torch
import numpy as np
from torch.utils.data import TensorDataset

from data_loading.dataloaders import split_dataset
from data_loading.utils import prepare_tone_dynamics
from utils.utils import set_seeds
from utils.visualise import plot_training_losses
from utils.audio import mel_to_audio, compare_mels, audio_to_mel
from models.synthesisModel import SynthesisModel, SynthesisLite
from models.synthesisTrainer import SynthesisTrainer


parser = argparse.ArgumentParser(
    description="Train an audio synthesizer on ECoG data."
)
# ----- I/O -------
parser.add_argument(
    '--sample_path', type=str, required=True,
    help='Path to the .mat file containing ECoG and audio samples.'
    ' The file should contain "ecog" and "audio" keys.'
)
parser.add_argument(
    '--figure_dir', type=str, default='figures',
    help='Directory to save the figures.'
)
parser.add_argument(
    '--audio_dir', type=str, default='audios',
    help='Directory to save the audio files.'
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


# TODO make these into a config file
non_discriminative_channels = [
    9,10,67,11,68,72,79,83,86,87,88,91,92,93,94,102,103,104,108,
    109,120,121,123,124,167,168,176,181,182,183,184,197,198,199,
    200,201,202,206,213,214,215,216,221,230,231,240
]

tone_dynamic_mapping = {
    1 : [4, 4, 4, 4, 4],
    2 : [2, 2.5, 3, 3.5, 4],
    3 : [2, 1.5, 1, 1.5, 2],
    4 : [5, 4, 3, 2, 1]
}

mel_kwargs = {
    'n_mels': 80,
    'n_fft': 2270,
    'hop_length': 567,
}


if __name__ == '__main__':
    params = parser.parse_args()

    if not os.path.exists(params.sample_path):
        raise FileNotFoundError(
            f"Data file '{params.sample_path}' does not exist.")
    
    if not os.path.exists(params.figure_dir):
        os.makedirs(params.figure_dir)

    if not os.path.exists(params.audio_dir):
        os.makedirs(params.audio_dir)

    if 'cuda' in params.device and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please use 'cpu' as device.")

    dataset = loadmat(params.sample_path)

    ecog_samples = dataset['ecog']
    ecog_samples = ecog_samples[:, non_discriminative_channels, :]
    audios = dataset['audio']

    mels = []
    for i, audio in enumerate(audios):
        mel = audio_to_mel(
            audio, params.audio_sampling_rate,
            mel_kwargs=mel_kwargs)
        
        mels.append(mel)
    
    mels = np.array(mels)  # shape (n_samples, n_mels * n_timepoints)
    
    print('Shape of Mel spectrogram:', mels.shape[1:])
    
    mels_dim = mels.shape[1]  # number of Mel coefficients

    # FIXME Using true labels for now, should use predicted labels
    tone_labels = dataset['tone'].flatten()
    syllable_labels = dataset['syllable'].flatten()

    if params.verbose > 0:
        print('Tones in the dataset ', np.unique(tone_labels))
        print('Syllables in the dataset ', np.unique(syllable_labels))

    tone_syllable_features = prepare_tone_dynamics(
        tone_dynamic_mapping, tone_labels, syllable_labels)
    print('Shape of tone-syllable features:', tone_syllable_features.shape[1:])

    n_samples, n_channels, n_timepoints = ecog_samples.shape

    if params.verbose > 0:
        print(
            f"Prepared {n_samples} ECoG samples with "
            f"shape {ecog_samples.shape[1:]}"
            f" and {tone_syllable_features.shape[1:]} "
            "dynamic features of syllable and tone labels.")

    ecog_samples = torch.tensor(ecog_samples, dtype=torch.float32)
    tone_syllable_features = torch.tensor(
        tone_syllable_features, dtype=torch.float32)
    mels = torch.tensor(mels, dtype=torch.float32)

    dataset = TensorDataset(ecog_samples, tone_syllable_features, mels)

    # repeated trainings with different splits
    mcds = []
    losses = []

    np.random.seed(params.seed)
    seeds = np.random.randint(0, 10000, params.repeat)

    for i, seed in enumerate(seeds):
        set_seeds(seed)

        train_loader, test_loader = split_dataset(
            dataset, params.train_ratio, params.batch_size,
            seed=seed
        )

        model = SynthesisLite(
            output_dim = mels_dim, n_channels=n_channels,
            n_timepoints=n_timepoints
        )

        trainer_verbose = params.verbose > 0 and i == 0
        trainer = SynthesisTrainer(
            model=model, device=params.device,
            learning_rate=params.lr, verbose=trainer_verbose
        )

        if params.verbose > 0:
            print(f"Training synthesizer with seed {seed}...")

        history = trainer.train(
            train_loader, params.epochs, verbose=params.verbose > 1)

        mcd, recon_mels, origin_mels = trainer.evaluate(test_loader)

        mcds.append(mcd)

        if params.verbose > 0:
            print(
                f"Finished trial {i+1} / {params.repeat}. "
                f"MCD: {mcd:.4f} dB"
            )
        
        losses.append([loss for loss, _ in history])
        
    mean_mcd = np.mean(mcds)
    std_mcd = np.std(mcds)

    print(f"-------- Training completed over {params.repeat} runs --------")
    print(
        f"MCD (Mel-Cepstral Distortion): {mean_mcd:.4f} dB ± {std_mcd:.4f} dB"
    )

    loss_figure_path = os.path.join(params.figure_dir, 'training_losses.png')
    plot_training_losses(losses, loss_figure_path)
    print("Saved training losses figure to ", loss_figure_path)

    n_samples = 10

    for i in range(n_samples):
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

        mel_fig_path = os.path.join(
            params.figure_dir, f'mel_{i}.png'
        )

        write_wave(audio_file_path, params.audio_sampling_rate, origin_wave)
        print("Saved original audio to ", audio_file_path)
        write_wave(recon_audio_file_path, params.audio_sampling_rate, recon_wave)
        print("Saved reconstructed audio to ", recon_audio_file_path)

        origin_mel = origin_mel.reshape(mel_kwargs['n_mels'], -1)
        recon_mel = recon_mel.reshape(mel_kwargs['n_mels'], -1)
        
        compare_mels(
            origin_mel, recon_mel, audio_sampling_rate=params.audio_sampling_rate,
            title1="Original Mel Spectrogram",
            title2="Synthesized Mel Spectrogram",
            file_path=mel_fig_path
            )

        print("Saved original mel spectrograms to ", mel_fig_path)
