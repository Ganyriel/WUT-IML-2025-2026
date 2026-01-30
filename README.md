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
- - **Note:** If you are using the Brave browser, please disable shields for the local link to work correctly.

## Running experiments:
- To experiment, train different model variants and compare the results use notebook Model/Experiments_Runner

## Exploratory data analysis:
- For examining obtained data use notebook EDA/exploratory_data_analysis

## Work division:

### Filip Filipkowski:

1. Researched available datasets
2. Implementated of the pipeline for obtaining initial data: all the scripts in DataPipelin directory, including downloading dataset, dividing the recordings into segments and directories,
and building manifest.
3. Prepared manual recordings of own voice for the possibility of better manual testing.
4. Implemented the logic of the train/validation/test split - the constraints to be satisfied are mentioned in the report
5. Added and tested some variants in the Experiments notebook, including LR scheduling, weight decay and MC dropout
6. Contributed to sections 2 and 4 of the report.
7. Connected app_gui with the final notebook

### Kuba Drażan:

1. Exploratory data analysis notebooks (exploratory_data_analysis, error_analysis)
2. I implemented engine for experiments, that tested several different models and saved results (Experiments_Runner)
3. Tested some variants in the Experiments notebook, including Baseline, Adam vs SGD
4. Wrote section 3 (Exploratory data analysis) of the report.
5. I implemented live demo example notebook and script shown at milestone 2, not gui (Live_demo)


### Héctor Rodon Llaberia & Iñaki Gutiérrez-Mantilla López:
1. Trained the final machine learning models used in the project.
2. Performed systematic comparison of different model architectures, hyperparameters and training strategies to identify the best-performing configuration.
3. Experimented with and evaluated multiple optimizers.
4. Selected and validated the final trained model based on quantitative results and generalization behavior.
5. Designed and implemented the graphical user interface (app_gui), integrating the trained model for user interaction and real-time inference.


### Filip Sabbo-Golebiowski
1. Created a file for creating spectrograms (not used in final version)
2. Added silence removal
3. Added uniformity of audio segments 
4. Did a large part of the coordination
5. Added some code for experimentation
6. Proofread most of the code
7. Researched for improvements and reasons for possible errors (e.g. data leaks)
8. Did some experimentation (hyper parameter tuning)
9. Wrote a first draft of the report

