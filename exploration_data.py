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
import utils


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
data = dt.fread('data/train.csv').to_pandas()
# %%
data = data.query('weight > 0')
# %%
ts_id_counts = data[['date', 'ts_id']].groupby('date').aggregate('count')

# %%
np.max(ts_id_counts)
# %%
np.min(ts_id_counts)
# %%
np.std(ts_id_counts)
# %%
np.mean(ts_id_counts)
# %%
np.median(ts_id_counts)
# %%
x = data['date'].unique()
# %%
y = ts_id_counts.values.reshape(-1)
# %%
plt.hist(y)
# %%
ts_id_dates_gt8000 = ts_id_counts[ts_id_counts['ts_id'] > 8000]
# %%
data_gt8000 = data[data['date'].isin(ts_id_dates_gt8000.index)]

# %%
mean_resp_weight = np.mean(data_gt8000['weight'])
# %%
missing_col_sums = data.isna().sum()
# %%
missing_cols_10per = data.loc[:, missing_col_sums > len(data)*0.1].columns

# %%
data = data.drop(missing_cols_10per, axis=1)
# %%
data['action'] = (data['weight'].values * data['resp'].values) > 0
# %%
data[data['action'] == True]
# %%
data['action_2'] = (data['resp']) > 0
# %%
data[data['action_2'] == True]
# %%
data.head()
# %%
