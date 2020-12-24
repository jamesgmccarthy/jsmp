# %%
from operator import index
import pickle
from plotly.subplots import SubplotXY
from tqdm import tqdm
from io import StringIO
from numpy.lib.arraysetops import unique
from sklearn.cluster import KMeans
import pandas as pd
import datatable as dt
import numpy as np
import matplotlib.pyplot as plt
import plotly as pl
import plotly.express as px
import seaborn as sns
from torch.nn.parallel.data_parallel import data_parallel
from torch.utils import data
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader, Subset, dataloader
from torchvision import transforms
from group_time_split import GroupTimeSeriesSplit
from sklearn.metrics import roc_auc_score
import copy


def plot_timeseries(data, feature):
    fig = plt.figure(figsize=(10, 6))
    x = range(len(data))
    y = data[feature]
    plt.plot(x, y)
    plt.grid()


def plotly_timeseries(data, feature):
    fig = px.scatter(data, x='ts_id', y=feature)
    fig.show()


# %%


class FinData(Dataset):
    def __init__(self, data, target, mode='train', transform=None, cache_dir=None):
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
                    'target': self.target.iloc[index].values,
                    'data': self.data[index],
                    'date': self.date.iloc[index].values
                }
                return sample

    def __len__(self):
        return len(self.data)


# %%
def load_data(root_dir, mode):
    if mode == 'train':
        data = dt.fread(root_dir+'train.csv').to_pandas()
    elif mode == 'test':
        data = dt.fread(root_dir+'example_test.csv').to_pandas()
    elif mode == 'sub':
        data = dt.fread(root_dir+'example_sample_submission.csv').to_pandas()
    return data


def preprocess_data(data):
    data = data.query('weight > 0').reset_index(drop=True)
    data['action'] = ((data['resp'].values) > 0).astype('float32')
    features = [f'feature_{i}' for i in range(1, 130)]+['weight']
    for col in features:
        data[col].fillna(data[col].mean(), inplace=True)
    target = data['action']
    date = data['date']
    data = data[features]
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    return data, target, features, date


# %%
data = load_data('data/', mode='train')
data, target, features, date = preprocess_data(data)
dataset = FinData(data, target, features)
# %%
"""
corr_matrix = sample_data.corr()
corr_matrix.iloc[np.where(corr_matrix.resp.abs() > 0.01)].index.values
corr_resp = corr_matrix['resp'].copy(deep=True)
corr_resp.pop('resp')
corr_resp

# %%
corr_resp.drop(['date', 'weight', 'resp_1', 'resp_2', 'resp_3',
                'resp_4', 'ts_id', 'action'], axis=0, inplace=True)

"""
# %%


class Classifier(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(in_features=130, out_features=500)
        self.bn1 = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(in_features=500, out_features=1000)
        self.bn2 = nn.BatchNorm1d(1000)
        self.fc3 = nn.Linear(in_features=1000, out_features=1000)
        self.bn3 = nn.BatchNorm1d(1000)
        self.fc4 = nn.Linear(in_features=1000, out_features=500)
        self.bn4 = nn.BatchNorm1d(500)
        self.dp = nn.Dropout(0.2)
        self.output = nn.Linear(in_features=500, out_features=out_features)

    def forward(self, x):
        x = self.dp(F.leaky_relu(self.bn1(self.fc1(x))))
        x = self.dp(F.leaky_relu(self.bn2(self.fc2(x))))
        x = self.dp(F.leaky_relu(self.bn3(self.fc3(x))))
        x = self.dp(F.leaky_relu(self.bn4(self.fc4(x))))
        output = torch.sigmoid(self.output(x))
        return output

    def save_model(self, root_dir):
        pass


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight, nn.init.calculate_gain('leaky_relu'))
        m.bias.data.fill_(1)


# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%
epochs = 100
epoch_loss = {'train': np.inf, 'val': np.inf}
epoch_metric = {'train': -np.inf, 'val': -np.inf}
batch_size = 500
gts = GroupTimeSeriesSplit()

for i, (train_idx, val_idx) in enumerate(gts.split(data, groups=date)):
    model = Classifier(len(features), 1).to(device=device)
    # model.apply(init_weights)
    optim = torch.optim.Adam(model.parameters(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, patience=5, factor=0.1, min_lr=1e-7, eps=1e-08)
    loss_fn = nn.BCELoss()
    train_set, val_set = torch.utils.data.Subset(
        dataset, train_idx), torch.utils.data.Subset(dataset, val_idx)

    train_sampler = torch.utils.data.sampler.BatchSampler(
        torch.utils.data.sampler.SequentialSampler(train_set),
        batch_size=batch_size, drop_last=False)
    val_sampler = torch.utils.data.sampler.BatchSampler(
        torch.utils.data.sampler.SequentialSampler(val_set),
        batch_size=batch_size, drop_last=False)

    dataloaders = {'train': DataLoader(dataset, sampler=train_sampler, num_workers=6),
                   'val': DataLoader(dataset, sampler=val_sampler, num_workers=6)}
    best_metric = -np.inf
    es_counter = 0
    for e, epoch in enumerate(range(1, epochs + 1)):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_metric = 0.0
            num_samples = 0
            bar = tqdm(dataloaders[phase])
            for b, batch in enumerate(bar, 1):
                bar.set_description(f'Epoch {epoch} {phase}'.ljust(20))
                x = batch['data'].to(device).to(
                    torch.float32)
                x = x.reshape(x.size(1), -1)
                y = batch['target'].to(device).reshape(-1, 1)
                optim.zero_grad()
                # forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(x)
                    loss = loss_fn(preds, y)
                    # back prop + optimize if in train phase
                    if phase == 'train':
                        loss.backward()
                        optim.step()
                    if phase == 'eval':
                        scheduler.step(loss)
                running_loss += loss.item() * x.size(0)
                running_metric += roc_auc_score(
                    y.detach().cpu().numpy(), preds.detach().cpu().numpy())
                num_samples += x.size(0)
                bar.set_postfix(loss=f'{running_loss / num_samples:0.5f}',
                                metric=f'{running_metric/b:0.5f}')
            epoch_loss[phase] = running_loss/num_samples
            epoch_metric[phase] = running_metric/b
            if phase == 'val' and epoch_metric['val'] > best_metric:
                best_metric = epoch_metric['val']
                best_model_wts = copy.deepcopy(model.state_dict())
                model_file = f'models/nn_model_fold_{i}.pth'
                torch.save(best_model_wts, model_file)
                es_counter = 0

        es_counter += 1
        if es_counter > 10:
            print(
                f'Early Stopping limit reached. Best Model saved to {model_file}')
            print(f'Best Metric achieved: {best_metric}')
            break

# %%

# %%
"""
model_xgboost = xgb.XGBClassifier()
p = {
    'learning_rate': 0.05,
    'max_depth': 50,
    'verbosity': 3,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'random_state': 42,
    'tree_method': 'gpu_hist',
}
# %%
for i, (train_idx, val_idx) in enumerate(gts.split(data, groups=date)):
    train_x, train_y = data[train_idx], target[train_idx]
    val_x, val_y = data[val_idx], target[val_idx]
    d_tr = xgb.DMatrix(train_x, train_y)
    d_val = xgb.DMatrix(val_x, val_y)
    clf = xgb.train(p, d_tr, evals=[(d_val, 'eval')])
# %%
"""
"""
# %%
for i, (train_idx, val_idx) in enumerate(gts.split(data, groups=date)):
    train_set, val_set = torch.utils.data.Subset(
        dataset, train_idx), torch.utils.data.Subset(dataset, val_idx)

    train_sampler = torch.utils.data.sampler.BatchSampler(
        torch.utils.data.sampler.SequentialSampler(train_set),
        batch_size=batch_size, drop_last=False)
    val_sampler = torch.utils.data.sampler.BatchSampler(
        torch.utils.data.sampler.SequentialSampler(val_set),
        batch_size=batch_size, drop_last=False)

    dataloaders = {'train': DataLoader(dataset, sampler=train_sampler),
                   'val': DataLoader(dataset, sampler=val_sampler)}
    for epoch in range(epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            num_samples = 0
            bar = tqdm(dataloaders[phase])
            for batch in bar:
                bar.set_description(f'Epoch {epoch} {phase}'.ljust(20))
                x = batch['data'].to(device).to(
                    torch.float32)
                x = x.reshape(x.size(1), -1)
                y = batch['target'].to(device).reshape(-1, 1)
                optim.zero_grad()
                # forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(x)
                    loss = loss_fn(preds, y)
                    # back prop + optimize if in train phase
                    if phase == 'train':
                        loss.backward()
                        optim.step()
                    if phase == 'eval':
                        scheduler.step(loss)
                running_loss += loss.item() * x.size(0)
                num_samples += x.size(0)
                bar.set_postfix(loss=f'{running_loss / num_samples:0.1f}')
# %%
x[0]

# %%
preds.mean()

# %%
y
# %%
preds[0]
# %%
plot_timeseries(x)
"""
