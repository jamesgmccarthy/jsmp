import datatable as dt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, Subset, BatchSampler, SequentialSampler, DataLoader


class FinData(Dataset):
    def __init__(self, data, target, date, mode='train', transform=None, cache_dir=None):
        self.data = data
        self.target = target
        self.mode = mode
        self.transform = transform
        self.cache_dir = cache_dir
        self.date = date

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index.to_list()
        if self.transform:
            return self.transform(self.data.iloc[index].values)
        else:
            if type(index) is list:
                sample = {
                    'target': torch.Tensor(self.target.iloc[index].values),
                    'data': torch.FloatTensor(self.data[index]),
                    'date': torch.Tensor(self.date.iloc[index].values)
                }
                return sample
            else:
                sample = {
                    'target': torch.Tensor(self.target.iloc[index]),
                    'data': torch.FloatTensor(self.data[index]),
                    'date': torch.Tensor(self.date.iloc[index])
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
        print(preds.shape)
        print(target.shape)
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


def preprocess_data(data, scale=False, nn=False):
    data = data.query('weight > 0').reset_index(drop=True)
    data['action'] = ((data['resp'].values) > 0).astype('float32')
    features = [
                   col for col in data.columns if 'feature' in col and col != 'feature_0'] + ['weight']
    for col in features:
        data[col].fillna(data[col].mean(), inplace=True)
    target = data['action']
    date = data['date']
    data = data[features]
    if scale:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
    if not scale and nn:
        data = data.values
    return data, target, features, date


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
