import os
import librosa

base_path = r'data_recordings'
accepted_folder = os.path.join(base_path, 'accepted')
if os.path.exists(accepted_folder):
    for speaker_folder in os.listdir(accepted_folder):
        speaker_path = os.path.join(accepted_folder, speaker_folder)
        if os.path.isdir(speaker_path):
            file_count = 0
            minimum = 10.0
            maximum = 0.0
            sum = 0.0
            for file in os.listdir(speaker_path):
                if file.endswith(('.wav', '.mp3', '.flac')):
                    relative_path = os.path.join(base_path, 'accepted', speaker_folder, file)
                    # print(relative_path)
                    y, sr = librosa.load(relative_path)
                    audio_length = librosa.get_duration(y = y, sr = sr)
                    # print()
                    # print(audio_length)
                    if (audio_length < minimum):
                        minimum = audio_length
                    if (audio_length > maximum):
                        maximum = audio_length
                    sum += audio_length
                    file_count += 1
            print(f"{speaker_folder}: {file_count} files")
            print("Minumum: ", minimum) 
            print("Maximum : ", maximum) 
            print("Sum: ", sum)
            print()

rejected_folder = os.path.join(base_path, 'rejected')
if os.path.exists(rejected_folder):
    for speaker_folder in os.listdir(rejected_folder):
        speaker_path = os.path.join(rejected_folder, speaker_folder)
        if os.path.isdir(speaker_path):
            file_count = 0
            minimum = 10.0
            maximum = 0.0
            sum = 0.0
            for file in os.listdir(speaker_path):
                if file.endswith(('.wav', '.mp3', '.flac')):
                    relative_path = os.path.join(base_path, 'rejected', speaker_folder, file)
                    # print(relative_path)
                    y, sr = librosa.load(relative_path)
                    audio_length = librosa.get_duration(y = y, sr = sr)
                    # print()
                    # print(audio_length)
                    if (audio_length < minimum):
                        minimum = audio_length
                    if (audio_length > maximum):
                        maximum = audio_length
                    sum += audio_length
                    file_count += 1
            print(f"{speaker_folder}: {file_count} files")  
            print("Minumum: ", minimum) 
            print("Maximum : ", maximum) 
            print("Sum: ", sum)
            print()


#base_path = r'data_recordings/accepted/p226'