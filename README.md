# reproducibility-study
Original code to train model is from https://github.com/annguy/time-aware-pbpm

## Starting Optuna Dashboard
Starting Optuna dashboard with one this command:

optuna-dashboard sqlite:///optuna-a100.db

Since the models Tax+T-LSTM and Tax+CS+T-LSTM performed the same, some Optuna studies didn't run 100 trials. 


## Folder structure
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

