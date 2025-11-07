import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import IPython.display as ipd

# Example file
audio_file = 'Ozymandias.wav'

# Some parameters (i don't know how melspectrograms work)
hopSize = 128
n_fft = 2048

# Extracting info from audio file
audio_data, sampling_rate = librosa.load(audio_file, sr=44100)

# Copied from tutorial for melspectrograms
mel_signal = librosa.feature.melspectrogram(y=audio_data, sr=sampling_rate, hop_length=hopSize, 
 n_fft=n_fft)
spectrogram = np.abs(mel_signal)
power_to_db = librosa.power_to_db(spectrogram, ref=np.max)
plt.figure(figsize=(8, 7))
librosa.display.specshow(power_to_db, sr=sampling_rate, x_axis="time", y_axis="mel", cmap="magma", 
 hop_length=hopSize)
plt.colorbar(label="dB")
plt.title('Mel-Spectrogram (dB)', fontdict=dict(size=18))
plt.xlabel('Time', fontdict=dict(size=15))
plt.ylabel('Frequency', fontdict=dict(size=15))
plt.show()
