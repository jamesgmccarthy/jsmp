import optuna
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import Subset, BatchSampler, SequentialSampler, DataLoader

from purged_group_time_series import PurgedGroupTimeSeriesSplit
from utils import load_data, preprocess_data, FinData


class Classifier(pl.LightningModule):
    def __init__(self, input_size, output_size, trial: optuna.Trial, early_stopping=10, batch_size=500):
        super.__init__()
        dim_1 = trial.suggest_int('dim_1', 100, 500)
        dim_2 = trial.suggest_int('dim_2', 100, 1000)
        dim_3 = trial.suggest_int('dim_3', 100, 1000)
        dim_4 = trial.suggest_int('dim_4', 100, 500)
        act_func = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'tanh', 'sigmoid'])
        act_dict = {'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid}
        self.dropout_prob = trial.suggest_uniform('dropout', 0.01, 0.5)
        self.lr = trial.suggest_uniform('lr', 0.0005, 0.01)
        self.activation = act_dict[act_func]
        self.input_size = input_size
        self.output_size = output_size
        self.loss = nn.BCELoss()
        self.early_stopping = early_stopping
        self.batch_size = batch_size
        self.train_log = pd.DataFrame({'auc': [0], 'loss': [0]})
        self.val_log = pd.DataFrame({'auc': [0], 'loss': [0]})
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
            return torch.sigmoid(out)

        def training_step(self, batch):
            x, y = batch['data'], batch['target']
            x = x.view(x.size(1), -1)
            y = y.view(-1, 1)
            logits = self(x)
            loss = self.loss(input=logits,
                             target=y)
            return loss

        def validation_step(self, batch):
            x, y = batch['data'], batch['target']
            x = x.view(x.size(1), -1)
            y = y.view(-1, 1)
            logits = self(x)
            loss = self.loss(input=logits,
                             target=y)
            return loss

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            return optimizer


def optimize(trial):
    gts = PurgedGroupTimeSeriesSplit(n_splits=5, group_gap=31)
    input_size = data.shape[0]
    output_size = 1

    for i, (train_idx, val_idx) in enumerate(gts.split(data, groups=date)):
        model = Classifier(input_size, output_size, trial, early_stopping=10)
        # model.apply(init_weights)
        train_set, val_set = Subset(
            dataset, train_idx), Subset(dataset, val_idx)
        train_sampler = BatchSampler(SequentialSampler(
            train_set), batch_size=batch_size, drop_last=False)
        val_sampler = BatchSampler(SequentialSampler(
            val_set), batch_size=batch_size, drop_last=False)
        dataloaders = {'train': DataLoader(dataset, sampler=train_sampler, num_workers=6),
                       'val':   DataLoader(dataset, sampler=val_sampler, num_workers=6)}
        trainer = pl.Trainer()
        trainer.fit(model, dataloaders['train'])
        # model.validation_step(dataloaders['val'])
        return trainer.ev


data = load_data(root_dir='./data/', mode='train')
data, target, features, date = preprocess_data(data, scale=True)
input_size = data.shape[0]
output_size = 1
dataset = FinData(data, target, features)
batch_size = 500
study = optuna.create_study(direction='minimize')
