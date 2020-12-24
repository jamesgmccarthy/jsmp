# %%
import xgboost as xgb
import lightgbm as lgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import datatable as dt
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.preprocessing import StandardScaler
from xgboost import sklearn
from group_time_split import GroupTimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import gc
from hyperopt import fmin, Trials, hp, tpe
from hyperopt.pyll.base import scope
from joblib import dump, load


def load_data(root_dir, mode, overide=None):
    if overide:
        data = dt.fread(overide).to_pandas()
    elif mode == 'train':
        data = dt.fread(root_dir+'train.csv').to_pandas()
    elif mode == 'test':
        data = dt.fread(root_dir+'example_test.csv').to_pandas()
    elif mode == 'sub':
        data = dt.fread(root_dir+'example_sample_submission.csv').to_pandas()
    return data


def preprocess_data(data):
    # data = data.query('weight > 0').reset_index(drop=True)
    data['action'] = ((data['resp'].values) > 0).astype('float32')
    features = [
        col for col in data.columns if 'feature' in col and col != 'feature_0']+['weight']
    for col in features:
        data[col].fillna(data[col].mean(), inplace=True)
    target = data['action']
    date = data['date']
    data = data[features]
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    return data, target, features, date


def optimize(params):
    print(params)
    p = {
        'learning_rate': params['learning_rate'],
        'max_depth': params['max_depth'],
        'gamma': params['gamma'],
        'subsample': params['subsample'],
        'colsample_bytree': params['colsample_bytree'],
        'min_child_weight': params['min_child_weight'],
        'tree_method': 'gpu_hist',
        'objective': 'binary:logistic',
        'eval_metric': 'auc'
    }
    scores = []
    gts = GroupTimeSeriesSplit()
    for i, (tr_idx, val_idx) in enumerate(gts.split(data, groups=date)):
        x_tr, x_val = data[tr_idx], data[val_idx]
        y_tr, y_val = target[tr_idx], target[val_idx]
        d_tr = xgb.DMatrix(x_tr, y_tr)
        d_val = xgb.DMatrix(x_val, y_val)
        clf = xgb.train(p, d_tr, params['n_round'], [
                        (d_val, 'eval')], early_stopping_rounds=50, verbose_eval=True)
        val_pred = clf.predict(d_val)
        score = roc_auc_score(y_val, val_pred)
        print(f'Fold {i} ROC AUC:\t', score)
        scores.append(score)
        del clf, val_pred, d_tr, d_val, x_tr, x_val, y_tr, y_val, score
        rubbish = gc.collect()
    avg_score = np.mean(scores)
    print('Avg Score:', avg_score)
    return avg_score


# %%
data = load_data('data/', mode='train', overide='filtered_train.csv')
data, target, features, date = preprocess_data(data)

# %%
"""
params = {'learning_rate': [0.1, 0.05, 0.005],
          'max_depth': [10, 15, 20, 40],
          'gamma': [0.01, 0.001],
          'subsample': [0.5, 0.75, 0.25],
          'colsample_bytree': [0.99, 0.95, 0.9],
          'min_child_weight': [0.5, 0.35, 0.25]}
clf = XGBClassifier(n_estimators=10, tree_method='gpu_hist',
                    objective='binary:logistic', verbosity=1, n_jobs=-1)
grid = GridSearchCV(clf, params, verbose=3, scoring='roc_auc')
"""
param_space = {'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
               'max_depth': scope.int(hp.quniform('max_depth', 3, 11, 1)),
               'gamma': hp.uniform('gamma', 0, 10),
               'min_child_weight': hp.uniform('min_child_weight', 0, 10),
               'subsample': hp.uniform('subsample', 0.1, 1),
               'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1),
               'n_round': scope.int(hp.quniform('n_round', 50, 1000, 25)),
               }
trials = Trials()
hopt = fmin(fn=optimize,
            space=param_space,
            algo=tpe.suggest,
            max_evals=50,
            trials=trials)
print(hopt)
dump(trials, 'XGB_Hyper_Opt_Trials', compress=True)
