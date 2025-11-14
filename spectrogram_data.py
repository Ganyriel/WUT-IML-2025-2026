import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import IPython.display as ipd
import skimage.io


# Some parameters

# settings
hop_length = 512 # number of samples per time-step in spectrogram
n_mels = 128 # number of bins in spectrogram. Height of image
time_steps = 200 # number of time-steps. Width of image

def spec_plot(path):
    """
    Plots a melspectrogram 
    
    Args: 
        path - file in which the audio is stored

    Output: 
        a plotted melspectrogram

    """

    n_fft = 2048 # this is the number of samples in a window per fft
    hop_length = 512 # The amount of samples we are shifting after each fft

    
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

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def spec_image(path, out, hop_length, n_mels):
    """
    Creates and saves a melspectrogram into a file
    
    Args: 
        path - file in which the audio is stored
        out - name of the png

    Output: 
        a melspectrogram in a file

    """

    
    y, sr = librosa.load(path, offset=1.0, duration=10.0, sr=22050)
    

    # extract a fixed length window
    start_sample = 0 # starting at beginning
    length_samples = time_steps*hop_length
    window = y[start_sample:start_sample+length_samples]

    
    # use log-melspectrogram
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                            n_fft=hop_length*2, hop_length=hop_length)
    mels = np.log(mels + 1e-9) # add small number to avoid log(0)

    # min-max scale to fit inside 8-bit range
    img = scale_minmax(mels, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0) # put low frequencies at the bottom in image
    img = 255-img # invert. make black==more energy

    # save as PNG
    skimage.io.imsave(out, img)



# Example 
if __name__ == "__main__":
    # setting input and output
    path = librosa.util.example('nutcracker') 
    out = 'out.png'

    # Plot
    #spec_plot(path)
    
    # convert to PNG
    spec_image(path, out=out, hop_length=hop_length, n_mels=n_mels)
    print('wrote file', out)

