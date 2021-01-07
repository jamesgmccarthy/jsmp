# %%
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
from torch.nn.modules.activation import Sigmoid

from utils import load_data, preprocess_data, FinData, create_dataloaders
from purged_group_time_series import PurgedGroupTimeSeriesSplit
from torch.utils.data import Subset, BatchSampler, SequentialSampler, DataLoader
from pytorch_lightning import Callback
from pytorch_lightning.metrics.functional import auroc, f1
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
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
    def __init__(self, input_size, output_size, params=None,
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
            nn.BatchNorm1d(input_size),
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
        logits = logits.view(-1)
        loss = self.loss(input=logits,
                         target=y)
        logits = torch.sigmoid(logits)
        auc_metric = roc_auc_score(y_true=y.cpu().numpy(),
                                   y_score=logits.cpu().detach().numpy())
        self.log('train_auc', auc_metric, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch['data'], batch['target']
        y = y.view(-1)
        x = x.view(x.size(1), -1)
        logits = self(x)
        logits = logits.view(-1)
        loss = self.loss(input=logits,
                         target=y)
        logits = torch.sigmoid(logits)

        auc = roc_auc_score(y_true=y.cpu().numpy(),
                            y_score=logits.cpu().detach().numpy())

        return {'loss': loss, 'y': y, 'logits': logits, 'auc': auc}

    def validation_epoch_end(self, val_step_outputs):
        epoch_loss = torch.tensor([x['loss'] for x in val_step_outputs]).mean()
        epoch_auc = torch.tensor([x['auc'] for x in val_step_outputs]).mean()
        self.log('val_loss', epoch_loss, prog_bar=True)
        self.log('val_auc', epoch_auc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        epoch_loss = torch.tensor([x['loss'] for x in outputs]).mean()
        epoch_auc = torch.tensor([x['auc'] for x in outputs]).mean()
        return {'test_loss': epoch_loss, 'test_auc': epoch_auc}

    def predict(self, batch):
        self.eval()
        x, y = batch['data'], batch['target']
        x = x.view(x.size(1), -1)
        x = self(x)
        return torch.sigmoid(x.view(-1))

    def prediction_loop(self, dataloader, return_tensor=True):
        bar = tqdm(dataloader)
        preds = []
        for batch in bar:
            preds.append(self.predict(batch))
        if return_tensor:
            return torch.cat(preds, dim=0)
        else:
            return preds

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
    data, target, features, date = preprocess_data(data, nn=True)
    dataset = FinData(data=data, target=target, date=date)
    gts = PurgedGroupTimeSeriesSplit(n_splits=5, group_gap=31)
    batch_size = 5000
    input_size = data.shape[-1]
    output_size = 1
    tb_logger = pl_loggers.TensorBoardLogger('logs/')
    p = {'batch_size': 4500,
         'dim_1': 312,
         'dim_2': 657,
         'dim_3': 723,
         'dim_4': 349,
         'activation': nn.LeakyReLU,
         'dropout': 0.06364070747726647,
         'lr': 0.0005004290173704919}

    for i, (train_idx, val_idx) in enumerate(gts.split(data, groups=date)):
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            os.path.join('models/', "fold_{}".format(i)), monitor="val_auc")
        model = Classifier(input_size=input_size,
                           output_size=output_size, trial=None, params=p)
        dataloaders = create_dataloaders(
            dataset, indexes={'train': train_idx, 'val': val_idx}, batch_size=batch_size)
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
    data_ = load_data(root_dir='./data/', mode='train')
    data, target, features, date = preprocess_data(data_, nn=True)
    dataset = FinData(data=data, target=target, date=date)
    gts = PurgedGroupTimeSeriesSplit(n_splits=5, group_gap=31)
    batch_size = 5000
    input_size = data.shape[-1]
    output_size = 1
    p = {'batch_size': 4600,
         'dim_1': 230,
         'dim_2': 850,
         'dim_3': 780,
         'dim_4': 190,
         'activation': nn.LeakyReLU,
         'dropout': 0.017122456592972537,
         'lr': 0.00013131268366473552}
    for i, (train_idx, val_idx) in enumerate(gts.split(data, groups=date)):
        if i == 4:
            if load:
                model = Classifier.load_from_checkpoint(checkpoint_path=load, input_size=input_size,
                                                        output_size=output_size, params=p)
                trainer = pl.Trainer(max_epochs=3, gpus=1, precision=16)
                model.lr = model.lr / 100
                dataloaders = create_dataloaders(
                    dataset, indexes={'train': train_idx}, batch_size=batch_size)

                trainer.fit(model, train_dataloader=dataloaders['train'],
                            val_dataloaders=dataloaders['train'])
                # trainer.test(test_dataloaders=train_loader)
                return model, features
            else:
                checkpoint_callback = ModelCheckpoint(
                    dirpath='logs', monitor='val_auc', mode='max', save_top_k=1, period=10)
                model = Classifier(input_size=input_size,
                                   output_size=output_size, params=p)
                dataloaders = create_dataloaders(
                    dataset, indexes={'train': train_idx, 'val': val_idx}, batch_size=batch_size)
                es = EarlyStopping(monitor='val_auc',
                                   patience=10, min_delta=0.0005, mode='max')
                trainer = pl.Trainer(max_epochs=500,
                                     gpus=1,
                                     callbacks=[checkpoint_callback, es],
                                     precision=16)
                trainer.fit(
                    model, train_dataloader=dataloaders['train'], val_dataloaders=dataloaders['val'])
                preds = model.prediction_loop(dataloaders['val'])
                trainer.test(test_dataloaders=dataloaders['val'])
                out = data_.iloc[val_idx, :]
                out['preds'] = preds.detach().numpy()
                out['target'] = target[val_idx]
                out.to_csv('output.csv')
            return model


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

        preds = torch.sigmoid(model.forward(vals.view(1, -1))).item()
        sample_prediction_df.action = np.where(
            preds > 0.5, 1, 0).astype(int).item()
        env.predict(sample_prediction_df)


# %%


def main():
    # train_cross_val()
    model = final_train()
    # best_model_path = checkpoint.best_model_path
    # model, features = final_train(load=best_model_path)
    # test_model(model, features)
    return model


if __name__ == '__main__':
    model = main()

# %%
