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

## Work division:
- Filip Filipkowski:
       1.Researched available datasets
       2.Implementated of the pipeline for obtaining initial data: all the scripts in DataPipelin directory, including downloading dataset, dividing the recordings into segments and directories,
       and building manifest.
       3. Prepared manual recordings of own voice for the possibility of better manual testing.
       4. Implemented the logic of the train/validation/test split - the constraints to be satisfied are mentioned in the report
       5. Added and tested some variants in the Experiments notebook, including LR scheduling, weight decay and MC dropout
       6. Contributed to sections 2 and 4 of the report.
       7. Connected app_gui with the final notebook
