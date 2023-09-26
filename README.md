# reproducibility-study
Original code to train model is from https://github.com/annguy/time-aware-pbpm

Adapted version with bugfixes is in folder `time-aware-pbpm`.

## Bugfixes

### Hardcoded Dropout Rate in CSModel
In the class `CSModel` the dropout rate is set three times to 0.2. This was changed to `hparams[HP_DROPOUT]` to be able to changed.

### Backslashes to Forwardslashes in Datahandler
The path in the `Datahandler` class used backslashes, which are used in Windows. This was changed to forwardslashes work on Linux.

### One-hot Encoding
In the class `Preprocess` the dictionaries were changed to start with 0. Another change was made to the `y_a` vector. In the class `NextStep` the variable `i` was changed to start at 0 in the function `getSymbol`.

### Reproducibility
The model classes have a seed parameter, if no seed is passed the models will behave randomly. The seed of 42 is set in the class `HyperparameterTune` as well as in `main.py`. The environment variables are set in `main.py` and in the class `HyperparameterTune`.

## Optuna
Code to use Optuna is in folder `optuna`. The sqlite databases that contain the Optuna results are also in this folder.

## Starting Optuna Dashboard
Starting Optuna dashboard with one this command:

`optuna-dashboard sqlite:///optuna-a100.db`

Since the models Tax+T-LSTM and Tax+CS+T-LSTM performed the same, some Optuna studies didn't perform 100 trials.

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

## Grid Search
In the folder `gridsearch` the generated files with the Hyperparameter Tuning option are located. For the model Tax+CS+T-LSTM only partial results exist, due to matching results with model Tax+T-LSTM and very long runtimes. To get the numbers, you need to run the GridSearch.py from the linked repository.

## Charts
The code to generate the charts that were used in the thesis are in the folder `scripts_charts`.