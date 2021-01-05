import os
import joblib
from numpy.core.numeric import True_
from pytorch_lightning import callbacks
from pytorch_lightning.metrics.functional.classification import precision
import torch
import torch.nn as nn
import pytorch_lightning as pl
import pandas as pd
import datatable as dt
import numpy as np
import optuna
from utils import load_data, preprocess_data, FinData, weighted_mean
from purged_group_time_series import PurgedGroupTimeSeriesSplit
from torch.utils.data import Subset, BatchSampler, SequentialSampler, DataLoader
from pytorch_lightning import Callback
from pytorch_lightning.metrics.functional import auroc, f1
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from optuna.integration import PyTorchLightningPruningCallback
import datetime
import neptune
import neptunecontrib.monitoring.optuna as opt_utils
from sklearn.metrics import roc_auc_score
from lightning_nn import Classifier


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight, nn.init.calculate_gain('leaky_relu'))
        m.bias.data.fill_(1)


def create_param_dict(trial, trial_file=None):
    if trial:
        dim_1 = trial.suggest_int('dim_1', 100, 500)
        dim_2 = trial.suggest_int('dim_2', 500, 1000)
        dim_3 = trial.suggest_int('dim_3', 500, 1000)
        dim_4 = trial.suggest_int('dim_4', 100, 500)
        act_func = trial.suggest_categorical(
            'activation', ['relu', 'leaky_relu'])
        act_dict = {'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU,
                    'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid}
        act_func = act_dict[act_func]
        dropout = trial.suggest_uniform('dropout', 0.01, 0.1)
        lr = trial.suggest_uniform('lr', 0.00005, 0.005)
        p = {'dim_1': dim_1, 'dim_2': dim_2, 'dim_3': dim_3,
             'dim_4': dim_4, 'activation': act_func, 'dropout': dropout,
             'lr': lr}
    elif trial_file:
        p = joblib.load(trial_file).best_params
    return p


def optimize(trial: optuna.Trial):
    gts = PurgedGroupTimeSeriesSplit(n_splits=5, group_gap=31)
    batch_size = trial.suggest_int('batch_size', 1000, 5000)
    input_size = data.shape[-1]
    output_size = 1
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        os.path.join('models/', "trial_{}".format(trial.number)), monitor="val_auc", mode='max')
    logger = MetricsCallback()
    metrics = []
    sizes = []
    p = create_param_dict(trial)
    for i, (train_idx, val_idx) in enumerate(gts.split(data, groups=date)):
        model = Classifier(input_size, output_size, params=p)
        # model.apply(init_weights)
        train_set, val_set = Subset(
            dataset, train_idx), Subset(dataset, val_idx)
        train_sampler = BatchSampler(SequentialSampler(
            train_set), batch_size=batch_size, drop_last=False)
        val_sampler = BatchSampler(SequentialSampler(
            val_set), batch_size=batch_size, drop_last=False)
        dataloaders = {'train': DataLoader(dataset, sampler=train_sampler, num_workers=10, pin_memory=True),
                       'val': DataLoader(dataset, sampler=val_sampler, num_workers=10, pin_memory=True)}
        es = EarlyStopping(monitor='val_auc', patience=10,
                           min_delta=0.0005, mode='max')
        trainer = pl.Trainer(logger=False,
                             max_epochs=500,
                             gpus=1,
                             callbacks=[checkpoint_callback, logger, PyTorchLightningPruningCallback(
                                 trial, monitor='val_auc'), es],
                             precision=16)
        trainer.fit(
            model, train_dataloader=dataloaders['train'], val_dataloaders=dataloaders['val'])
        val_auc = logger.metrics[-1]['val_auc'].item()
        metrics.append(val_auc)
        sizes.append(len(train_idx))
    metrics_mean = weighted_mean(metrics, sizes)
    return metrics_mean


data = load_data(root_dir='./data/', mode='train')
data, target, features, date = preprocess_data(data, nn=True)
dataset = FinData(data=data, target=target, date=date)
api_token = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiYWQxMjg3OGEtMGI1NC00NzFmLTg0YmMtZmIxZjcxZDM2NTAxIn0='
neptune.init(api_token=api_token,
             project_qualified_name='jamesmccarthy65/JSMP')
nn_exp = neptune.create_experiment('NN_HPO')
nn_neptune_callback = opt_utils.NeptuneCallback(experiment=nn_exp)
pruner = optuna.pruners.MedianPruner()
study = optuna.create_study(direction='maximize', pruner=pruner)
study.optimize(optimize, n_trials=50, callbacks=[nn_neptune_callback])
joblib.dump(study, f'HPO/nn_hpo_{str(datetime.datetime.now().date())}.pkl')
