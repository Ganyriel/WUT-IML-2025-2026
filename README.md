# WUT-IML-2025-2026
Voice Classifier project for Introduction to Machine Learning 
 


## Obtaining data:
- To get the data run the script run_pipeline_to_get_data.py
- You will end up with zipped dataset file, and all the recordings in folder data_recordings, divide per accepted/rejected and per speaker
- In data_recordings there is also csv file with speaker id's, relative paths to recordings and labels


## Running the real time program:
- Open the MainProgram/main_notebook.ipynb
- Run all cells
- Open the local link from the last cell output
- If the model file is ready you can skip the cells that load data and train it

## Running experiments:
- To experiment, train different model variants and compare the results use notebook Model/Experiments_Runner

## Exploratory data analysis:
- For examining obtained data use notebook EDA/exploratory_data_analysis
