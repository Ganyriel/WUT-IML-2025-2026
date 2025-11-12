import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import IPython.display as ipd


# Some parameters (i don't quite know how melspectrograms work, i copied it from a tutorial)

n_fft = 2048 # this is the number of samples in a window per fft
hop_length = 512 # The amount of samples we are shifting after each fft

def spec(audio_file):
    """
    Input: audio file

    Output: a plotted melspectrogram

    """

    # Extracting info from audio file
    audio_data, sampling_rate = librosa.load(audio_file)

    # Copied from tutorial for melspectrograms
    mel_signal = librosa.feature.melspectrogram(y=audio_data, sr=sampling_rate, hop_length=hop_length, 
    n_fft=n_fft)
    spectrogram = np.abs(mel_signal)
    power_to_db = librosa.power_to_db(spectrogram, ref=np.max)
    plt.figure(figsize=(8, 7))
    librosa.display.specshow(power_to_db, sr=sampling_rate, x_axis="time", y_axis="mel", cmap="magma", 
    hop_length=hop_length,fmax=sampling_rate/2)
    plt.colorbar(label="dB")
    plt.title('Mel-Spectrogram (dB)', fontdict=dict(size=18))
    plt.xlabel('Time', fontdict=dict(size=15))
    plt.ylabel('Frequency', fontdict=dict(size=15))
    plt.show()


# Example 
audio_file = 'Ozymandias.wav'
spec(audio_file)




