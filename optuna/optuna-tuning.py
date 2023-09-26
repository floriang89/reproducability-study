import optuna
from optuna.samplers import TPESampler
from tensorboard.plugins.hparams import api as hp
from src.Models.TLSTM_layer import TLSTM_layer
from src.Data.Datahandler import Datahandler
from src.Features.Preprocess import Preprocess
from src.Features.ComputeCW import ComputeCW
from src.Models.Model import *
from src.Models.Test import NextStep
from datetime import date
from datetime import datetime
import os
import time

class Objective:
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model

    def __call__(self, trial):
        F = Datahandler()
        name=F.read_data(self.dataset)

        spamreader,max_task = F.log2np()
        D=Preprocess()
        divisor,divisor2,divisor3 = D.divisor_cal(spamreader)
        maxlen,chars,target_chars,char_indices,indices_char,target_char_indices,target_indices_char= D.dict_cal()
        num_features = len(chars)+5
        X,y_a,y_t,d_t=D.training_set(num_features)

        cw = ComputeCW()
        class_weights = cw.compute_class_weight(F.spamread)
        print('class_weights are: ', class_weights)

        if self.model == 'CS':
            M = CSModel(maxlen,
                max_task,
                target_chars,
                name,
                num_features)
        elif self.model == 'TLSTM':
            M=ALL_TLSTM_Model(maxlen,
                                max_task,
                                target_chars,
                                name,
                                num_features)
        elif self.model == 'CSTLSTM':
            M = CS_TLSTM_Model(maxlen,
                                max_task,
                                target_chars,
                                name,
                                num_features)

        num_units_trial = trial.suggest_int("num_units", 10, 150, step=5)
        HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([num_units_trial]))
        dropout_trial = trial.suggest_float("dropout", 0.0, 1.0)
        HP_DROPOUT = hp.HParam('dropout', hp.Discrete([dropout_trial]))
        optimizer_trial = trial.suggest_categorical("optimizer", ["nadam", "adam", "rmsprop"])
        HP_OPTIMIZER = optimizer = hp.HParam('optimizer', hp.Discrete([optimizer_trial]))
        learning_rate_trial = trial.suggest_float("learning_rate", 1e-5, 2e-1, log=True)
        HP_LEARNING_RATE = learning_rate = hp.HParam('learning_rate', hp.Discrete([learning_rate_trial]))
        
        hparams = {
            HP_NUM_UNITS:HP_NUM_UNITS.domain.values[0],
            HP_DROPOUT:HP_DROPOUT.domain.values[0],
            HP_OPTIMIZER:HP_OPTIMIZER.domain.values[0],
            HP_LEARNING_RATE:HP_LEARNING_RATE.domain.values[0],
        }
        run_stat={h.name: hparams[h] for h in hparams}
        run_name = str(run_stat.values())
    
        print(run_stat)
        if self.model == 'CS':
            val_loss, history = M.train(X, y_a, y_t, class_weights, hparams,
                    HP_NUM_UNITS, HP_DROPOUT,HP_OPTIMIZER, HP_LEARNING_RATE)
        elif self.model == 'TLSTM':
            val_loss, history = M.train(X,d_t,y_a,y_t, hparams,HP_NUM_UNITS,HP_DROPOUT,
                                        HP_OPTIMIZER,HP_LEARNING_RATE)
        elif self.model == 'CSTLSTM':
            val_loss, history = M.train(X, d_t, y_a, y_t, class_weights,
                                    hparams, HP_NUM_UNITS, HP_DROPOUT,
                                    HP_OPTIMIZER, HP_LEARNING_RATE)

        return val_loss

os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = 'true'

datasets = {'bpi_12_w.csv'}
models = {'CSTLSTM'}

i = 1
for dataset in datasets:
    for model in models:
        total_runs = len(datasets) * len(models)
        
        log = (f"##########################################\n"
                f"  Run {i} of {total_runs}\n"   
                f"  Dataset:\t{dataset}\n"
                f"  Model:\t{model}\n"
                f"  Start:\t{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"##########################################")
        print(log)
        i +=1
        if (model == 'TLSTM' or model == 'CSTLSTM') and dataset == 'bpi_12_w.csv':
            batch_size = 512
        else:
            batch_size = 64
        study_name = '{}_{}_bsize_{}_{}'.format(dataset, model, batch_size, '2023-09-19')
        storage_name = "sqlite:///{}.db".format("optuna")
        print("Study name: ", study_name)
        sampler = TPESampler(seed=42)
        # optuna.delete_study(study_name=study_name, storage=storage_name)
        study = optuna.create_study(sampler=sampler, study_name=study_name, storage=storage_name, direction="minimize", load_if_exists=True)
        study.optimize(Objective(dataset, model), n_trials=20)