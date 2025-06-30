import librosa
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def audio_to_mel(
        audio: np.ndarray, audio_sampling_rate: int,
        mel_in_db: bool=True,
        mel_kwargs: Optional[dict]=None
    ) -> np.ndarray:
    """
    Convert audio signal to Mel spectrogram.

    Parameters
    ----------
    audio : np.ndarray
        Input audio signal of shape (n_samples,).
    audio_sampling_rate : int
        Sample rate of the audio signal.
    mel_in_db : bool
        If True, return Mel spectrogram in dB scale.
        If False, return Mel spectrogram in linear scale.
    mel_kwargs : Optional[dict]
        Additional keyword arguments for `librosa.feature.melspectrogram`.
        e.g. `n_mels`, `hop_length`, `fmin`, `fmax`, etc.

    Returns
    -------
    np.ndarray
        Mel spectrogram of shape (n_mels * n_timepoints, ).
    """
    if len(audio.shape) > 1:
        raise ValueError("Audio input must be a 1D array.")

    mel = librosa.feature.melspectrogram(
        y=audio, sr=audio_sampling_rate, **mel_kwargs
    )

    if mel_in_db:
        mel = librosa.power_to_db(mel, ref=np.max)
    
    return mel.reshape(-1)


def mel_to_audio(
        mel: np.ndarray, n_mels: int,
        audio_sampling_rate: int=24414,
        mel_in_db: bool=True,
        **kwargs
    ) -> np.ndarray:
    """
    Restore audio waveform from mel spectrogram
    using Griffin-Lim algorithm.

    Parameters
    ----------
    mel : np.ndarray
        Mel spectrogram to restore audio from.
        Shape (n_mels * n_timepoints,).
    n_mels : int
        Number of mel frequency bins.
    audio_sampling_rate : int, optional
        Sample rate of the audio to restore.
        Default is 24414 Hz.
    **kwargs
        Additional keyword arguments to pass to
        `librosa.feature.inverse.mel_to_audio`.

    Returns
    -------
    np.ndarray
        Restored audio waveform of shape 
        (n_samples,).
    """
    mel = mel.reshape(n_mels, -1)  # (n_timepoints, n_mels)

    if mel_in_db:  # convert back to power scale
        mel = librosa.db_to_power(mel, ref=0.0001)

    wave = librosa.feature.inverse.mel_to_audio(
        mel,
        sr=audio_sampling_rate,
        **kwargs
    )

    return wave

def visualise_mel(
        mel : np.ndarray,
        audio_sampling_rate: int=24414,
        mel_in_db: bool=True,
        file_path: Optional[str]=None,
        show: bool=True
    ) -> None:
    """
    Visualise mel spectrogram using librosa.

    Parameters
    ----------
    mel : np.ndarray
        Mel spectrogram to visualise.
        Shape (n_mels, n_timepoints).
    audio_sampling_rate : int, optional
        Sample rate of the audio to visualise.
        Default is 24414 Hz.
    mel_in_db : bool, optional
        Whether the mel spectrogram is in dB scale.
        Default is True.
    file_path : Optional[str], optional
        File path to save the visualisation.
        e.g. "mel_spectrogram.png".
        If None, the plot will be shown instead.
        Default is None.
    show : bool, optional
        Whether to show the plot.
        Default is True.
    """

    if show:
        plt.figure(figsize=(10, 4))

    librosa.display.specshow(
        mel, sr=audio_sampling_rate, x_axis='time',
        y_axis='mel', cmap='coolwarm'
    )
    if mel_in_db:
        plt.colorbar(format='%+2.0f dB')
    else:
        plt.colorbar(format='%+2.0f')

    if show:
        plt.title('Mel Spectrogram', fontsize=18)
        plt.tight_layout()
        if file_path:
            plt.savefig(file_path, dpi=400)
            plt.close()
        else:
            plt.show()

def compare_mels(
        mel1: np.ndarray, mel2: np.ndarray,
        audio_sampling_rate: int=24414,
        title1: str='Mel Spectrogram 1',
        title2: str='Mel Spectrogram 2',
        mel_in_db: bool=True,
        file_path: Optional[str]=None
    ) -> None:
    """
    Compare two mel spectrograms side by side.

    Parameters
    ----------
    mel1 : np.ndarray
        First mel spectrogram to compare.
        Shape (n_mels, n_timepoints).
    mel2 : np.ndarray
        Second mel spectrogram to compare.
        Shape (n_mels, n_timepoints).
    audio_sampling_rate : int, optional
        Sample rate of the audio to visualise.
        Default is 24414 Hz.
    title1 : str, optional
        Title for the first mel spectrogram.
        Default is 'Mel Spectrogram 1'.
    title2 : str, optional
        Title for the second mel spectrogram.
        Default is 'Mel Spectrogram 2'.
    mel_in_db : bool, optional
        Whether the mel spectrograms are in dB scale.
        Default is True.
    file_path : Optional[str], optional
        File path to save the visualisation.
        e.g. "mel_comparison.png".
        If None, the plot will be shown instead.
        Default is None.
    """

    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    visualise_mel(mel1, audio_sampling_rate, mel_in_db, show=False)
    plt.title(title1)

    plt.subplot(1, 2, 2)
    visualise_mel(mel2, audio_sampling_rate, mel_in_db, show=False)
    plt.title(title2)

    if file_path:
        plt.savefig(file_path, dpi=400)
        plt.close()
    else:
        plt.show()
