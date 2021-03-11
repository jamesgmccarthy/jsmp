import os
import random

import datatable as dt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, Subset, BatchSampler, SequentialSampler, DataLoader


# from lightning_nn import Classifier


class FinData(Dataset):
    def __init__(self, data, target, date, mode='train', transform=None, cache_dir=None, multi=False):
        self.data = data
        self.target = target
        self.mode = mode
        self.transform = transform
        self.cache_dir = cache_dir
        self.date = date
        self.multi = multi

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index.to_list()
        if self.transform:
            return self.transform(self.data.iloc[index].values)
        else:
            if type(index) is list:
                if self.multi == False:
                    sample = {
                        'target': torch.Tensor(self.target.iloc[index].values),
                        'data':   torch.FloatTensor(self.data[index]),
                        'date':   torch.Tensor(self.date.iloc[index].values)
                    }
                elif self.multi == True:
                    sample = {
                        'target': torch.Tensor(self.target[index]),
                        'data':   torch.FloatTensor(self.data[index]),
                        'date':   torch.Tensor(self.date.iloc[index].values)
                    }

            else:
                if self.multi == False:
                    sample = {
                        'target': torch.Tensor(self.target.iloc[index]),
                        'data':   torch.FloatTensor(self.data[index]),
                        'date':   torch.Tensor(self.date.iloc[index])
                    }
                elif self.multi == True:
                    sample = {
                        'target': torch.Tensor(self.target[index]),
                        'data':   torch.FloatTensor(self.data[index]),
                        'date':   torch.Tensor(self.date.iloc[index])
                    }
        return sample

    def __len__(self):
        return len(self.data)


def linear_combination(x, y, epsilon):
    print(x)
    print(y)
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)


def load_data(root_dir, mode, overide=None):
    if overide:
        data = dt.fread(overide).to_pandas()
    elif mode == 'train':
        data = dt.fread(root_dir + 'train.csv').to_pandas()
    elif mode == 'test':
        data = dt.fread(root_dir + 'example_test.csv').to_pandas()
    elif mode == 'sub':
        data = dt.fread(root_dir + 'example_sample_submission.csv').to_pandas()
    return data


def preprocess_data(data: pd.DataFrame, scale: bool = False, nn: bool = False,
                    action: str = 'weight'):
    """
    Preprocess the data.

    Parameters
    ----------
    data
        Pandas DataFrame
    scale
        scale data with unit std and 0 mean
    nn
        return data as np.array
    missing
        options to replace missing data with - mean, median, 0
    action
        options to create action value  - weight = (weight * resp) > 0
                                        - combined = (resp_cols) > 0
                                        - multi = each resp cols >0

    Returns
    -------
    """

    data = data.query('weight > 0').reset_index(drop=True)
    data = data.query('date > 85').reset_index(drop=True)
    if action == 'weight':
        data['action'] = (data['resp'].values > 0).astype('float32')
    if action == 'multi':
        resp_cols = ['resp', 'resp_1', 'resp_2', 'resp_3', 'resp_4']
        for i in range(len(resp_cols)):
            data['action_' + str(i)] = (data['weight'] *
                                        data[resp_cols[i]] > 0).astype('int')
    features = [col for col in data.columns if 'feature' in col] + ['weight']
    date = data['date']

    if action == 'multi':
        target = np.array([data['action_' + str(i)]
                           for i in range(len(resp_cols))]).T

    else:
        target = data['action']
    data = data[features]
    if scale:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
    if not scale and nn:
        data = data.values
    return data, target, features, date


def calc_data_mean(array, cache_dir=None, fold=None, train=True, mode='mean'):
    if train:
        if mode == 'mean':
            f_mean = np.nanmean(array, axis=0)
            if cache_dir and fold:
                np.save(f'{cache_dir}/f_{fold}_mean.npy', f_mean)
            elif cache_dir:
                np.save(f'{cache_dir}/f_mean.npy', f_mean)
            array = np.nan_to_num(array) + np.isnan(array) * f_mean
        if mode == 'median':
            f_med = np.nanmedian(array, axis=0)
            if cache_dir and fold:
                np.save(f'{cache_dir}/f_{fold}_median.npy', f_med)
            elif cache_dir:
                np.save(f'{cache_dir}/f_median.npy', f_med)
            array = np.nan_to_num(array) + np.isnan(array) * f_med
        if mode == 'zero':
            array = np.nan_to_num(array) + np.isnan(array) * 0
    if not train:
        if mode == 'mean':
            f_mean = np.load(f'{cache_dir}/f_mean.npy')
            array = np.nan_to_num(array) + np.isnan(array) * f_mean
        if mode == 'median':
            f_med = np.load(f'{cache_dir}/f_med.npy')
            array = np.nan_to_num(array) + np.isnan(array) * f_med
        if mode == 'zero':
            array = np.nan_to_num(array) + np.isnan(array) * 0
    return array


def weighted_mean(scores, sizes):
    largest = np.max(sizes)
    weights = [size / largest for size in sizes]
    return np.average(scores, weights=weights)


def create_dataloaders(dataset: Dataset, indexes: dict, batch_size):
    train_idx = indexes.get('train', None)
    val_idx = indexes.get('val', None)
    test_idx = indexes.get('test', None)
    dataloaders = {}
    if train_idx:
        train_set = Subset(
            dataset, train_idx)
        train_sampler = BatchSampler(SequentialSampler(
            train_set), batch_size=batch_size, drop_last=False)
        dataloaders['train'] = DataLoader(
            dataset, sampler=train_sampler, num_workers=10, pin_memory=True)
    if val_idx:
        val_set = Subset(dataset, val_idx)
        val_sampler = BatchSampler(SequentialSampler(
            val_set), batch_size=batch_size, drop_last=False)
        dataloaders['val'] = DataLoader(
            dataset, sampler=val_sampler, num_workers=10, pin_memory=True)
    if test_idx:
        test_set = Subset(dataset, test_idx)
        test_sampler = BatchSampler(SequentialSampler(
            test_set), batch_size=batch_size, drop_last=False)
        dataloaders['test'] = DataLoader(
            dataset, sampler=test_sampler, num_workers=10, pin_memory=True)
    return dataloaders


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_model(path, input_size, output_size, p, pl_lightning):
    if os.path.isdir(path):
        models = []
        for file in os.listdir(path):
            if pl_lightning:
                model = Classifier.load_from_checkpoint(checkpoint_path=file, input_size=input_size,
                                                        output_size=output_size, params=p)
            else:
                model = Classifier(input_size, output_size, params=p)
                model.load_state_dict(torch.load(file))
            models.append(model)
        return models
    elif os.path.isfile(path):
        if pl_lightning:
            return Classifier.load_from_checkpoint(checkpoint_path=path, input_size=input_size,
                                                   output_size=output_size, params=p)
        else:
            model = Classifier(input_size, output_size, params=p)
            model.load_state_dict(torch.load(path))
            return model


def init_weights(m, func):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight, nn.init.calculate_gain(func))


def read_api_token(file='api.txt'):
    with open(file, 'r') as f:
        api = f.readline()
    return api
