# %%
import datatable as dt
import pandas as pd
import numpy as np
from purged_group_time_series import PurgedGroupTimeSeriesSplit
import utils
import csv
# %%
train = utils.load_data('data/', 'train')
# %%
data, target, features, date = utils.preprocess_data(train, nn=False)
# %%
gts = PurgedGroupTimeSeriesSplit(n_splits=5, group_gap=31)
for i, (train_idx, val_idx) in enumerate(gts.split(data, groups=date)):
    df = pd.DataFrame({'indexes': train_idx})
    df.to_csv(f'indices/train_{i}.csv')
    df = pd.DataFrame({'val': val_idx})
    df.to_csv(f'indices/val_{i}.csv')
# %%
fold_1 = dt.fread('train_idx_0.csv').to_pandas()
# %%
fold_1.T
# %%
