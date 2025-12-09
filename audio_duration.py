import os
import librosa

# path to the data recordings folder
base_path = r'data_recordings'

for ac in ['accepted', 'rejected']: 
    folder = os.path.join(base_path, ac)
    if os.path.exists(folder):
        for speaker_folder in os.listdir(folder):
            speaker_path = os.path.join(folder, speaker_folder)
            if os.path.isdir(speaker_path):

                # Initializing trackers
                file_count = 0
                minimum = 100.0
                maximum = 0.0
                sum = 0.0

                for file in os.listdir(speaker_path):
                    if file.endswith(('.wav', '.mp3', '.flac')):
                        relative_path = os.path.join(base_path, ac, speaker_folder, file)

                        # Loading audio data and checking length
                        y, sr = librosa.load(relative_path)
                        audio_length = librosa.get_duration(y = y, sr = sr)

                        # Keeping track of minimum, maximum, sum and number of audio segments
                        if (audio_length < minimum):
                            minimum = audio_length
                        if (audio_length > maximum):
                            maximum = audio_length
                        sum += audio_length
                        file_count += 1

                # Printing
                print(f"{speaker_folder}: {file_count} files")
                print("Minumum: ", minimum) 
                print("Maximum : ", maximum) 
                print("Sum: ", sum)
                print()
