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
from utils import load_data, preprocess_data, FinData
from purged_group_time_series import PurgedGroupTimeSeriesSplit
from torch.utils.data import Subset, BatchSampler, SequentialSampler, DataLoader
from pytorch_lightning import Callback
from pytorch_lightning.metrics.functional import auroc, f1
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from optuna.integration import PyTorchLightningPruningCallback
import datetime
import neptune
import neptunecontrib.monitoring.optuna as opt_utils


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


class Classifier(pl.LightningModule):
    def __init__(self, input_size, output_size, trial: optuna.Trial, params, early_stopping=10, batch_size=500,
                 model_path='models/'):
        super(Classifier, self).__init__()
        dim_1 = params['dim_1']
        dim_2 = params['dim_2']
        dim_3 = params['dim_3']
        dim_4 = params['dim_4']
        self.dropout_prob = params['dropout']
        self.lr = params['lr']
        self.activation = params['act_func']
        self.input_size = input_size
        self.output_size = output_size
        self.loss = nn.BCEWithLogitsLoss()
        self.batch_size = batch_size
        self.train_log = pd.DataFrame({'auc': [0], 'loss': [0]})
        self.val_log = pd.DataFrame({'auc': [0], 'loss': [0]})
        self.model_path = model_path
        self.encoder = nn.Sequential(
            nn.Linear(input_size, dim_1),
            nn.BatchNorm1d(dim_1),
            self.activation(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(dim_1, dim_2),
            nn.BatchNorm1d(dim_2),
            self.activation(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(dim_2, dim_3),
            nn.BatchNorm1d(dim_3),
            self.activation(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(dim_3, dim_4),
            nn.BatchNorm1d(dim_4),
            self.activation(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(dim_4, self.output_size)
        )

    def forward(self, x):
        out = self.encoder(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch['data'], batch['target']
        x = x.view(x.size(1), -1)
        y = y.view(-1)
        logits = self(x)
        logits = torch.sigmoid(logits).view(-1)
        loss = self.loss(input=logits,
                         target=y)
        pbar = {'t_auc': auroc(pred=logits, target=y)}
        return {'loss': loss, 'progress_bar': pbar}

    def validation_step(self, batch, batch_idx):
        x, y = batch['data'], batch['target']
        x = x.view(x.size(1), -1)
        y = y.view(-1)
        logits = self(x)
        logits = torch.sigmoid(logits).view(-1)
        loss = self.loss(input=logits,
                         target=y)
        auc = auroc(pred=logits, target=y)
        pbar = {'t_auc': auc}
        return {'loss': loss, 'auc': auc, 'progress_bar': pbar}

    def validation_epoch_end(self, val_step_outputs):
        epoch_loss = torch.tensor([x['loss'] for x in val_step_outputs]).mean()
        epoch_auc = torch.tensor([x['auc'] for x in val_step_outputs]).mean()
        pbar = {'val_loss': epoch_loss,
                'val_auc': epoch_auc}
        self.log('val_auc', epoch_auc)
        return {'val_loss': epoch_loss, 'val_auc': epoch_auc, 'progress_bar': pbar}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.1, min_lr=1e-7, eps=1e-08
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_auc'}


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight, nn.init.calculate_gain('leaky_relu'))
        m.bias.data.fill_(1)


def create_param_dict(trial, trial_file=None):
    if trial:
        dim_1 = trial.suggest_int('dim_1', 100, 500)
        dim_2 = trial.suggest_int('dim_2', 100, 1000)
        dim_3 = trial.suggest_int('dim_3', 100, 1000)
        dim_4 = trial.suggest_int('dim_4', 100, 500)
        act_func = trial.suggest_categorical(
            'activation', ['relu', 'leaky_relu'])
        act_dict = {'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU,
                    'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid}
        act_func = act_dict[act_func]
        dropout = trial.suggest_uniform('dropout', 0.01, 0.5)
        lr = trial.suggest_uniform('lr', 0.0005, 0.01)
        p = {'dim_1': dim_1, 'dim_2': dim_2, 'dim_3': dim_3,
             'dim_4': dim_4, 'act_func': act_func, 'dropout': dropout,
             'lr': lr}
    elif trial_file:
        p = joblib.load(trial_file).best_params
    return p


def optimize(trial):
    gts = PurgedGroupTimeSeriesSplit(n_splits=5, group_gap=31)
    batch_size = trial.suggest_int('batch_size', 500, 1500)
    input_size = data.shape[-1]
    output_size = 1
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        os.path.join('models/', "trial_{}".format(trial.number)), monitor="val_auc"
    )
    logger = MetricsCallback()
    #p = create_param_dict(trial)
    p = {'dim_1': 496	, 'dim_2': 385	, 'dim_3': 143,	 'dim_4': 100,
         'activation': 'relu',	 'dropout': 0.49934040768390675, 'lr': 0.0006661013327594837}
    for i, (train_idx, val_idx) in enumerate(gts.split(data, groups=date)):
        model = Classifier(input_size, output_size, trial, params=p)
        # model.apply(init_weights)
        train_set, val_set = Subset(
            dataset, train_idx), Subset(dataset, val_idx)
        train_sampler = BatchSampler(SequentialSampler(
            train_set), batch_size=batch_size, drop_last=False)
        val_sampler = BatchSampler(SequentialSampler(
            val_set), batch_size=batch_size, drop_last=False)
        dataloaders = {'train': DataLoader(dataset, sampler=train_sampler, num_workers=10, pin_memory=True),
                       'val': DataLoader(dataset, sampler=val_sampler, num_workers=10, pin_memory=True)}
        es = EarlyStopping(monitor='val_auc', patience=10, min_delta=0.0005)
        trainer = pl.Trainer(logger=False,
                             checkpoint_callback=checkpoint_callback,
                             max_epochs=500,
                             gpus=1,
                             callbacks=[logger, PyTorchLightningPruningCallback(
                                 trial, monitor='val_auc'), es],
                             precision=16)
        trainer.fit(
            model, train_dataloader=dataloaders['train'], val_dataloaders=dataloaders['val'])
    print(logger.metrics[-1]['val_auc'].item())
    return logger.metrics[-1]['val_auc'].item()


data = load_data(root_dir='./data/', mode='train')
data, target, features, date = preprocess_data(data, scale=True)
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
