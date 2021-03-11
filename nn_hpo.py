import datetime
import os
import copy
import joblib
import neptune
import neptunecontrib.monitoring.optuna as opt_utils
import optuna
import pytorch_lightning as pl
import torch.nn as nn
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import Subset, BatchSampler, SequentialSampler, DataLoader
import torch
import numpy as np
from resnet import ResNet as Classifier
from purged_group_time_series import PurgedGroupTimeSeriesSplit
from utils import load_data, preprocess_data, FinData, read_api_token, weighted_mean, seed_everything, calc_data_mean, \
    create_dataloaders


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
    if trial and not trial_file:
        dim_1 = trial.suggest_int('dim_1', 50, 300)
        dim_2 = trial.suggest_int('dim_2', 200, 500)
        dim_3 = trial.suggest_int('dim_3', 200, 1000)
        dim_4 = trial.suggest_int('dim_4', 200, 500)
        dim_5 = trial.suggest_int('dim_5', 50, 300)
        act_func = trial.suggest_categorical(
            'activation', ['relu', 'leaky_relu', 'gelu'])
        act_dict = {'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU,
                    'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'gelu': nn.GELU}
        act_func = act_dict[act_func]
        dropout = trial.suggest_uniform('dropout', 0.2, 0.5)
        lr = trial.suggest_uniform('lr', 0.00005, 0.005)
        label_smooth = trial.suggest_uniform('label_smoothing', 0.001, 0.1)
        weight_decay = trial.suggest_uniform('weight_decay', 0.005, 0.05)
        amsgrad = trial.suggest_categorical('amsgrad', [False, True])
        p = {'dim_1': dim_1, 'dim_2': dim_2, 'dim_3': dim_3,
             'dim_4': dim_4, 'dim_5': dim_5, 'activation': act_func, 'dropout': dropout,
             'lr': lr, 'label_smoothing': label_smooth, 'weight_decay': weight_decay,
             'amsgrad': amsgrad}
    elif trial and trial_file:
        p = joblib.load(trial_file).best_params
        if not p.get('dim_5', None):
            p['dim_5'] = 75
        if not p.get('label_smoothing', None):
            p['label_smoothing'] = 0.094
        act_dict = {'relu': nn.ReLU,
                    'leaky_relu': nn.LeakyReLU, 'gelu': nn.GELU}
        act_func = trial.suggest_categorical(
            'activation', ['leaky_relu', 'gelu'])
        p['activation'] = act_dict[p['activation']]
        p['weight_decay'] = trial.suggest_uniform('weight_decay', 0.005, 0.05)
        p['amsgrad'] = trial.suggest_categorical('amsgrad', [False, True])
    return p


def optimize(trial: optuna.Trial, data_dict):
    gts = PurgedGroupTimeSeriesSplit(n_splits=5, group_gap=10)
    input_size = data_dict['data'].shape[-1]
    output_size = 5
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        os.path.join('models/', "trial_resnet_{}".format(trial.number)), monitor="val_auc", mode='max')
    logger = MetricsCallback()
    metrics = []
    sizes = []
    # trial_file = 'HPO/nn_hpo_2021-01-05.pkl'
    trial_file = None
    p = create_param_dict(trial, trial_file)
    p['batch_size'] = trial.suggest_int('batch_size', 8000, 15000)
    for i, (train_idx, val_idx) in enumerate(gts.split(data_dict['data'], groups=data_dict['date'])):
        idx = np.concatenate([train_idx, val_idx])
        data = copy.deepcopy(data_dict['data'][idx])
        target = copy.deepcopy(data_dict['target'][idx])
        date = copy.deepcopy(data_dict['date'][idx])
        train_idx = [i for i in range(0, max(train_idx) + 1)]
        val_idx = [i for i in range(len(train_idx), len(idx))]
        data[train_idx] = calc_data_mean(
            data[train_idx], './cache', train=True, mode='mean')
        data[val_idx] = calc_data_mean(
            data[val_idx], './cache', train=False, mode='mean')
        model = Classifier(input_size, output_size, params=p)
        # model.apply(init_weights)
        dataset = FinData(data=data, target=target, date=date, multi=True)
        dataloaders = create_dataloaders(
            dataset, indexes={'train': train_idx, 'val': val_idx}, batch_size=p['batch_size'])
        es = EarlyStopping(monitor='val_loss', patience=10,
                           min_delta=0.0005, mode='min')
        trainer = pl.Trainer(logger=False,
                             max_epochs=500,
                             gpus=1,
                             callbacks=[checkpoint_callback, logger, PyTorchLightningPruningCallback(
                                 trial, monitor='val_loss'), es],
                             precision=16)
        trainer.fit(
            model, train_dataloader=dataloaders['train'], val_dataloaders=dataloaders['val'])
        val_loss = logger.metrics[-1]['val_loss'].item()
        metrics.append(val_loss)
        sizes.append(len(train_idx))
    metrics_mean = weighted_mean(metrics, sizes)
    return metrics_mean


def main():
    seed_everything(0)
    data = load_data(root_dir='./data/', mode='train')
    data, target, features, date = preprocess_data(
        data, nn=True, action='multi')

    api_token = read_api_token()
    neptune.init(api_token=api_token,
                 project_qualified_name='jamesmccarthy65/JSMP')
    nn_exp = neptune.create_experiment('Resnet_HPO_Multiclass')
    nn_neptune_callback = opt_utils.NeptuneCallback(experiment=nn_exp)
    study = optuna.create_study(direction='minimize')
    data_dict = {'data': data, 'target': target,
                 'features': features, 'date': date}
    study.optimize(lambda trial: optimize(trial, data_dict=data_dict), n_trials=100,
                   callbacks=[nn_neptune_callback])
    joblib.dump(study, f'HPO/nn_hpo_{str(datetime.datetime.now().date())}.pkl')


if __name__ == '__main__':
    main()
