# %%
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, accuracy_score, recall_score, precision_score, roc_auc_score
import joblib
import optuna
import pandas as pd
from sklearn.metrics._plot.roc_curve import plot_roc_curve
from torch.utils.data import dataloader
import utils
# %%
nn = joblib.load('HPO/nn_hpo_2021-01-05.pkl')

# %%
optuna.visualization.plot_param_importances(nn)

# %%
optuna.visualization.plot_parallel_coordinate(nn)
# %%
# %%
optuna.visualization.plot_slice(nn)
# %%
nn.best_params
# %%
# %%
data = utils.load_data(root_dir='./data/', mode='train')
data_nn, target, features, date = utils.preprocess_data(data, nn=True)
data_, target, features, date = utils.preprocess_data(data)
# %%
data[data['ts_id'] == 1981286]
# %%
data = pd.read_csv('output.csv')
# %%
preds = data['preds']

# %%
"""
acc = accuracy_score(y_pred=preds, y_true=data['target'])
rec = recall_score(y_pred=preds, y_true=data['target'])
prec = precision_score(y_pred=preds, y_true=data['target'])
"""
auc = roc_auc_score(y_true=data['target'], y_score=data['preds'])
# %%
print('acc', acc)
print('rec', rec)
print('prec', prec)
print('auc', auc)

# %%
df['preds']
# %%
auc
# %%
