import copy
import os

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

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
        y = y.view(y.size(1), -1)
        logits = self(x)
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
        y = y.view(y.size(1), -1)
        logits = self(x)
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
    data_, target_, features, date = preprocess_data(
        data_, nn=True, action='multi')
    input_size = data_.shape[-1]
    output_size = target_.shape[-1]
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
        data[train_idx] = calc_data_mean(
            data[train_idx], './cache', train=True, mode='mean')
        data[val_idx] = calc_data_mean(
            data[val_idx], './cache', train=False, mode='mean')
        dataset = FinData(data=data, target=target, date=date, multi=True)
        dataloaders = create_dataloaders(
            dataset, indexes={'train': train_idx, 'val': val_idx}, batch_size=p['batch_size'])
        es = EarlyStopping(monitor='val_auc', patience=10,
                           min_delta=0.0005, mode='max')
        trainer = pl.Trainer(logger=tb_logger,
                             max_epochs=1,
                             gpus=1,
                             callbacks=[checkpoint_callback, es],
                             precision=16)
        trainer.fit(
            model, train_dataloader=dataloaders['train'], val_dataloaders=dataloaders['val'])
        torch.save(model.state_dict(), f'models/fold_{i}_state_dict.pth')
        models.append(model)
    return models, features


def fillna_npwhere(array, values):
    if np.isnan(array.sum()):
        array = np.nan_to_num(array) + np.isnan(array) * values
    return array


def test_model(models, features, cache_dir='cache'):
    env = janestreet.make_env()
    iter_test = env.iter_test()
    if type(models) == list:
        models = [model.eval() for model in models]
    else:
        models.eval()
    f_mean = np.load(f'{cache_dir}/f_mean.npy')
    for (test_df, sample_prediction_df) in tqdm(iter_test):
        if test_df['weight'].item() > 0:
            vals = torch.FloatTensor(
                fillna_npwhere(test_df[features].values, f_mean))
            if type(models) == list:
                # calc mean of each models prediction of each response rather than mean of all predicted responses by each model
                preds = [torch.sigmoid(model.forward(vals.view(1, -1))).detach().numpy()
                         for model in models]
                pred = np.mean(np.mean(preds, axis=0))
            else:
                pred = torch.sigmoid(models.forward(vals.view(1, -1))).item()
            sample_prediction_df.action = np.where(
                pred > 0.5, 1, 0).astype(int).item()
        else:
            sample_prediction_df.action = 0
        env.predict(sample_prediction_df)


def main():
    p = {'dim_1': 167,
         'dim_2': 454,
         'dim_3': 371,
         'dim_4': 369,
         'dim_5': 155,
         'activation': nn.LeakyReLU,
         'dropout': 0.21062362698532755,
         'lr': 0.0022252024054478523,
         'label_smoothing': 0.05564974140461841,
         'weight_decay': 0.04106097088288333,
         'amsgrad': True,
         'batch_size': 10072}
    models, features = cross_val(p)
    test_model(models, features)


if __name__ == '__main__':
    main()
