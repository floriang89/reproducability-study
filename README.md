# reproducibility-study
Original code to train model is from https://github.com/annguy/time-aware-pbpm

## Starting Optuna Dashboard
Starting Optuna dashboard with one this command:

`optuna-dashboard sqlite:///optuna-a100.db`

Since the models Tax+T-LSTM and Tax+CS+T-LSTM performed the same, some Optuna studies didn't run 100 trials. 


## Folder structure models and results
Each folder contains the used model, the evaluation file, the generated charts for accuracy and MAE are included. 

### Models
1 = Tax+CS

2 = Tax+T-LSTM

3 = Tax+CS+T-LSTM

### Bugfix
no_ohe_fix = one-hot encoding was not fix

with_ohe_fix = one-hot encoding was fixed

### Datasets
bpi = bpi_12_w.csv 

helpdesk = helpdesk.csv

### Hyperparameter Tuning approach
optuna = Optuna

gridsearch = Grid search

## Folder Structure Grid Search
In this folder the generated files with the Hyperparameter Tuning option are located. For the model Tax+CS+T-LSTM 
only partial results exist, due to matching results with model Tax+T-LSTM and very long runtimes. To get the 
numbers, you need to run the GridSearch.py from the linked repository. 
