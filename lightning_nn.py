from numba import njit
import os
import joblib
from numpy.core.numeric import True_
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.metrics.functional.classification import precision
import torch
from torch._C import dtype
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
from pytorch_lightning import loggers as pl_loggers
import janestreet
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


class Classifier(pl.LightningModule):
    def __init__(self, input_size, output_size, trial: optuna.Trial = None, params=None,
                 model_path='models/'):
        super(Classifier, self).__init__()
        dim_1 = params['dim_1']
        dim_2 = params['dim_2']
        dim_3 = params['dim_3']
        dim_4 = params['dim_4']
        self.dropout_prob = params['dropout']
        self.lr = params['lr']
        self.activation = params['activation']
        self.input_size = input_size
        self.output_size = output_size
        self.loss = nn.BCEWithLogitsLoss()

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
        y = y.view(-1)
        x = x.view(x.size(1), -1)
        logits = self(x)
        logits = torch.sigmoid(logits).view(-1)
        loss = self.loss(input=logits,
                         target=y)
        auc_metric = roc_auc_score(y_true=y.cpu().numpy(),
                                   y_score=logits.cpu().detach().numpy())
        self.log('train_auc', auc_metric, on_step=False, on_epoch=True)
        pbar = {'t_auc': auc_metric}
        return {'loss': loss, 'progress_bar': pbar}

    def validation_step(self, batch, batch_idx):
        x, y = batch['data'], batch['target']
        y = y.view(-1)
        x = x.view(x.size(1), -1)
        logits = self(x)
        logits = torch.sigmoid(logits).view(-1)
        loss = self.loss(input=logits,
                         target=y)
        auc = roc_auc_score(y_true=y.cpu().numpy(),
                            y_score=logits.cpu().detach().numpy())
        pbar = {'v_auc': auc}
        return {'loss': loss, 'auc': auc, 'progress_bar': pbar}

    def validation_epoch_end(self, val_step_outputs):
        epoch_loss = torch.tensor([x['loss'] for x in val_step_outputs]).mean()
        epoch_auc = torch.tensor([x['auc'] for x in val_step_outputs]).mean()
        pbar = {'val_loss': epoch_loss,
                'val_auc': epoch_auc}
        self.log('val_auc', epoch_auc)
        return {'val_loss': epoch_loss, 'val_auc': epoch_auc, 'progress_bar': pbar}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        epoch_loss = torch.tensor([x['loss'] for x in outputs]).mean()
        epoch_auc = torch.tensor([x['auc'] for x in outputs]).mean()
        return {'test_loss': epoch_loss, 'test_auc': epoch_auc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.1, min_lr=1e-7, eps=1e-08
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_auc'}


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight, nn.init.calculate_gain('relu'))
        m.bias.data.fill_(1)


def train_cross_val():
    data = load_data(root_dir='./data/', mode='train')
    data, target, features, date = preprocess_data(data, scale=True)
    dataset = FinData(data=data, target=target, date=date)
    gts = PurgedGroupTimeSeriesSplit(n_splits=5, group_gap=31)
    batch_size = 1108
    input_size = data.shape[-1]
    output_size = 1
    tb_logger = pl_loggers.TensorBoardLogger('logs/')
    p = {'dim_1': 496, 'dim_2': 385, 'dim_3': 143, 'dim_4': 100,
         'activation': nn.ReLU,
         'dropout': 0.49934040768390675,
         'lr': 0.0006661013327594837}

    for i, (train_idx, val_idx) in enumerate(gts.split(data, groups=date)):
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            os.path.join('models/', "fold_{}".format(i)), monitor="val_auc")
        model = Classifier(input_size=input_size,
                           output_size=output_size, trial=None, params=p)
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
        trainer = pl.Trainer(logger=tb_logger,
                             checkpoint_callback=checkpoint_callback,
                             max_epochs=500,
                             gpus=1,
                             callbacks=[es],
                             precision=16)
        trainer.fit(
            model, train_dataloader=dataloaders['train'], val_dataloaders=dataloaders['val'])
        results = trainer.test(test_dataloaders=dataloaders['val'])
        print(results)


def final_train(load=False):
    data = load_data(root_dir='./data/', mode='train')
    data, target, features, date = preprocess_data(data, scale=True)
    dataset = FinData(data=data, target=target, date=date)
    gts = PurgedGroupTimeSeriesSplit(n_splits=5, group_gap=31)
    batch_size = 1108
    input_size = data.shape[-1]
    output_size = 1
    p = {'dim_1': 496, 'dim_2': 385, 'dim_3': 143, 'dim_4': 100,
         'activation': nn.ReLU,
         'dropout': 0.49934040768390675,
         'lr': 0.0006661013327594837}
    for i, (train_idx, val_idx) in enumerate(gts.split(data, groups=date)):
        if i == 4:
            if load:
                model = Classifier(input_size=input_size,
                                   output_size=output_size, params=p)
                trainer = pl.Trainer(
                    max_epochs=26, gpus=1, resume_from_checkpoint=load)
                model.lr = model.lr/100
                train_set = Subset(dataset, val_idx)
                train_sampler = BatchSampler(SequentialSampler(
                    train_set), batch_size=batch_size, drop_last=False)
                train_loader = DataLoader(
                    dataset, sampler=train_sampler, num_workers=10, pin_memory=True)
                trainer.fit(model, train_dataloader=train_loader,
                            val_dataloaders=train_loader)
                return model, features
            else:
                checkpoint_callback = ModelCheckpoint(
                    dirpath='logs', monitor='val_auc', mode='max', save_top_k=1, period=10)
                model = Classifier(input_size=input_size,
                                   output_size=output_size, trial=None, params=p)
                train_set, val_set = Subset(
                    dataset, train_idx), Subset(dataset, val_idx)
                train_sampler = BatchSampler(SequentialSampler(
                    train_set), batch_size=batch_size, drop_last=False)
                val_sampler = BatchSampler(SequentialSampler(
                    val_set), batch_size=batch_size, drop_last=False)
                dataloaders = {'train': DataLoader(dataset, sampler=train_sampler, num_workers=10, pin_memory=True),
                               'val': DataLoader(dataset, sampler=val_sampler, num_workers=10, pin_memory=True)}
                es = EarlyStopping(monitor='val_auc',
                                   patience=10, min_delta=0.0005, mode='max')
                trainer = pl.Trainer(max_epochs=500,
                                     gpus=1,
                                     callbacks=[checkpoint_callback, es],
                                     precision=16)
                trainer.fit(
                    model, train_dataloader=dataloaders['train'], val_dataloaders=dataloaders['val'])
            trainer.test(test_dataloaders=dataloaders['val'])
            return model, checkpoint_callback, features


@njit
def fillna_npwhere_njit(array, values):
    if np.isnan(array.sum()):
        array = np.where(np.isnan(array), values, array)
    return array


def test_model(model, features):
    model.eval()
    env = janestreet.make_env()
    iter_test = env.iter_test()
    for (test_df, sample_prediction_df) in tqdm(iter_test):
        vals = torch.FloatTensor(
            fillna_npwhere_njit(test_df[features].values, 0.0))

        preds = torch.sigmoid(model.forward(vals.view(1, -1)))
        sample_prediction_df.action = (preds > 0.5).to(dtype=int)
        env.predict(sample_prediction_df)


def main():
    # train_cross_val()
    model, checkpoint, features = final_train()
    best_model_path = checkpoint.best_model_path
    model, features = final_train(load=best_model_path)
    test_model(model, features)


main()
