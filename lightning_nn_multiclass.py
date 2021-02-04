import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from numba import njit
from pytorch_lightning import Callback
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import janestreet
from purged_group_time_series import PurgedGroupTimeSeriesSplit
from utils import load_data, preprocess_data, FinData, create_dataloaders, calc_data_mean, init_weights


class Classifier(pl.LightningModule):
    def __init__(self, input_size, output_size, params, model_path='models/'):
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
        self.weight_decay = params['weight_decay']
        self.amsgrad = params['amsgrad']
        self.label_smoothing = params['label_smoothing']
        self.model_path = model_path
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, dim_1, bias=False),
            nn.BatchNorm1d(dim_1),
            self.activation(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(dim_1, dim_2, bias=False),
            nn.BatchNorm1d(dim_2),
            self.activation(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(dim_2, dim_3, bias=False),
            nn.BatchNorm1d(dim_3),
            self.activation(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(dim_3, dim_4, bias=False),
            nn.BatchNorm1d(dim_4),
            self.activation(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(dim_4, self.output_size, bias=False)
        )

    def forward(self, x):
        out = self.encoder(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch['data'], batch['target']
        x = x.view(x.size(1), -1)
        logits = self(x)
        logits = logits.view(-1)
        loss = self.loss(input=logits, target=y)
        logits = torch.sigmoid(logits)
        auc_metric = roc_auc_score(y_true=y.cpu().numpy(),
                                   y_score=logits.cpu().detach().numpy())
        self.log('train_auc', auc_metric, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch['data'], batch['target']
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
        self.log('test_loss', epoch_loss)
        self.log('test_auc', epoch_auc)

    def configure_optimizers(self):
        # weight_decay = self.weight_decay,
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,
                                     amsgrad=self.amsgrad)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.1, min_lr=1e-7, eps=1e-08
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_auc'}


def cross_val(p) -> object:
    data_ = load_data(root_dir='./data/', mode='train')
    data_, target_, features, date = preprocess_data(data_, nn=True, action='multi')
    input_size = data_.shape[-1]
    output_size = target_.shape[0]
    gts = PurgedGroupTimeSeriesSplit(n_splits=5, group_gap=5)
    models = []
    tb_logger = pl_loggers.TensorBoardLogger('logs/multiclass_')
    for i, (train_idx, val_idx) in enumerate(gts.split(data_, groups=date)):
        idx = np.concatenate([train_idx, val_idx])
        data = copy.deepcopy(data_[idx])
        target = copy.deepcopy(target_[idx])
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            os.path.join('models/', 'multi_class_fold_{}'.format(i)), monitor='val_auc', save_top_k=1, period=10,
            mode='max'
        )
        model = Classifier(input_size, output_size, p)
        if p['activation'] == nn.ReLU:
            model.apply(lambda m: init_weights(m, 'relu'))
        elif p['activation'] == nn.LeakyReLU:
            model.apply(lambda m: init_weights(m, 'leaky_relu'))
        train_idx = [i for i in range(0, max(train_idx) + 1)]
        val_idx = [i for i in range(len(train_idx), len(idx))]
        data[train_idx] = calc_data_mean(data[train_idx], './cache', train=True, mode='mean')
        data[val_idx] = calc_data_mean(data[val_idx], './cache', train=False, mode='mean')
        dataset = FinData(data=data, target=target, date=date)
        dataloaders = create_dataloaders(
            dataset, indexes={'train': train_idx, 'val': val_idx}, batch_size=p['batch_size'])
        es = EarlyStopping(monitor='val_auc', patience=10,
                           min_delta=0.0005, mode='max')
        trainer = pl.Trainer(logger=tb_logger,
                             max_epochs=5,
                             gpus=1,
                             callbacks=[checkpoint_callback, es],
                             precision=16)
        trainer.fit(
            model, train_dataloader=dataloaders['train'], val_dataloaders=dataloaders['val'])
        torch.save(model.state_dict(), f'models/fold_{i}_state_dict.pth')
        models.append(model)
    return models, features


def main():
    p = {'batch_size':   4986, 'dim_1': 248, 'dim_2': 487,
         'dim_3':        269, 'dim_4': 218, 'dim_5': 113,
         'activation':   nn.ReLU, 'dropout': 0.01563457578202565,
         'lr':           0.00026372556533974916, 'label_smoothing': 0.06834918091900156,
         'weight_decay': 0.005270589494631074, 'amsgrad': False}
    models, features = cross_val(p)


if __name__ == '__main__':
    main()
