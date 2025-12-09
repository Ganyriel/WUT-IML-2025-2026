# WUT-IML-2025-2026
Wut project for Introduction to Machine Learning 

# TODO
- don't remove silence for train set?
- remove absolute silence? (we only remove relative silence, i.e. parts which are % more silent than rest)
- delete one of the manifests
- check if no data leaks or similar issues appear, since results are too good
- Clean training_model.ipynb (includes adding comments, putting functions in one place instead of them being scattered throughout the code)
  

## Files:
- ### Preprocessing:
  - spectrogram_data.py: creates and plots melspectrograms
  - audio_duration.py: analyzes the duration of all audio files (after segmentation)
  - TODO Add more files descriptions here
- ### Training:
  - trained_model.ipynb 


## Obtaining data:
- To get the data run the script run_pipeline_to_get_data.py
- You will end up with zipped dataset file, and all the recordings in folder data_recordings, divide per accepted/rejected and per speaker
- In data_recordings there is also csv file with speaker id's, relative paths to recordings and labels


## Notes
- usage of melspectrograms is mandatory 
- data exploration: we should avoid data leaks
- data exploration: we should delete silent 'accepted' audio segments
