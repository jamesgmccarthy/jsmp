# %%
import joblib
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
import optuna
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
    data = data.query('weight > 0').reset_index(drop=True)
    data['action'] = ((data['resp'].values) > 0).astype('float32')
    data = data.query('date > 80').reset_index(drop=True)
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


def optimize(trial):
    p = {'learning_rate': trial.suggest_uniform('learning_rate', 1e-5, 1e-2),
         'max_depth': trial.suggest_int('max_depth', 5, 30),
         'max_leaves': trial.suggest_int('max_leaves', 5, 50),
         'subsample': trial.suggest_uniform('subsample', 0.3, 1.0),
         'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.3, 1.0),
         'boosting': trail.suggest_categorical('boosting',),
         'objective': 'binary:logistic',
         'booster': 'gbtree',
         'tree_method': 'gpu_hist',
         'verbosity': 1,
         'n_jobs': 10,
         'eval_metric': 'auc'}
    print('Choosing parameters:', p)
    scores = []
    gts = GroupTimeSeriesSplit()
    for i, (tr_idx, val_idx) in enumerate(gts.split(data, groups=date)):
        x_tr, x_val = data[tr_idx], data[val_idx]
        y_tr, y_val = target[tr_idx], target[val_idx]
        d_tr = xgb.DMatrix(x_tr, y_tr)
        d_val = xgb.DMatrix(x_val, y_val)
        clf = xgb.train(p, d_tr, 1000, [
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
"""
print('creating XGBoost Trials')
study = optuna.create_study(direction='maximize')
study.optimize(optimize, n_trials=50)
joblib.dump(study, 'xgb_hpo.pkl')


def loptimize(trial):
    p = {'learning_rate': trial.suggest_uniform('learning_rate', 1e-5, 1e-2),
         'max_leaves': trial.suggest_int('max_leaves', 5, 100),
         'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.3, 1.0),
         'feature_fraction': trial.suggest_uniform('feature_fraction', 0.3, 1.0),
         'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.3, 1.0),
         'objective': 'binary',
         'boosting_type': 'gbdt',
         'verbose': 1,
         'metric': 'auc'}
    scores = []
    gts = GroupTimeSeriesSplit()
    for i, (tr_idx, val_idx) in enumerate(gts.split(data, groups=date)):
        x_tr, x_val = data[tr_idx], data[val_idx]
        y_tr, y_val = target[tr_idx], target[val_idx]
        train = lgb.Dataset(x_tr, label=y_tr)
        val = lgb.Dataset(x_val, label=y_val)
        clf = lgb.train(p, train, 1000, valid_sets=[
                        val], early_stopping_rounds=50, verbose_eval=True)
        preds = clf.predict(x_val)
        score = roc_auc_score(y_val, preds)
        print(f'Fold {i} ROC AUC:\t', score)
        scores.append(score)
        del clf, preds, train, val, x_tr, x_val, y_tr, y_val, score
        rubbish = gc.collect()
    avg_score = np.mean(scores)
    print('Avg Score:', avg_score)
    return avg_score


print('Creating LightGBM Trials')
study = optuna.create_study(direction='maximize')
study.optimize(loptimize, n_trials=50)
joblib.dump(study, 'lgb_hpo.pkl')
"""
gts = GroupTimeSeriesSplit()
for i, (tr_idx, val_idx) in enumerate(gts.split(data, groups=date)):
    if i == 4:
        p = {'learning_rate': 0.009931637628758162,
             'max_leaves': 72,
             'bagging_fraction': 0.3030113880887123,
             'feature_fraction': 0.8,
             'objective': 'binary',
             'boosting_type': 'gbdt',
             'verbose': 1,
             'metric': 'auc'}
        x_tr, x_val = data[tr_idx], data[val_idx]
        y_tr, y_val = target[tr_idx], target[val_idx]
        train = lgb.Dataset(x_tr, label=y_tr)
        val = lgb.Dataset(x_val, label=y_val)
        clf = lgb.LGBMClassifier(n_estimators=1000, verbose_eval=True, **p)
        clf.fit(x_tr, y_tr, early_stopping_rounds=50)
        preds = clf.predict(x_val)
        score = roc_auc_score(y_val, preds)
        print(f'Fold {i} ROC AUC:\t', score)
