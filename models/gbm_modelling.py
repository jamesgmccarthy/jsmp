# %%
import datetime
import gc

import joblib
import lightgbm as lgb
import neptune
import neptunecontrib.monitoring.optuna as opt_utils
import optuna
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from purged_group_time_series import PurgedGroupTimeSeriesSplit
from utils.utils import read_api_token, weighted_mean, load_data, preprocess_data


def optimize(trial: optuna.trial.Trial):
    p = {'learning_rate': trial.suggest_uniform('learning_rate', 1e-4, 1e-1),
         'max_depth': trial.suggest_int('max_depth', 5, 30),
         'max_leaves': trial.suggest_int('max_leaves', 5, 50),
         'subsample': trial.suggest_uniform('subsample', 0.3, 1.0),
         'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.3, 1.0),
         'min_child_weight': trial.suggest_int('min_child_weight', 5, 100),
         'lambda': trial.suggest_uniform('lambda', 0.05, 0.2),
         'alpha': trial.suggest_uniform('alpha', 0.05, 0.2),
         'objective': 'binary:logistic',
         'booster': 'gbtree',
         'tree_method': 'gpu_hist',
         'verbosity': 1,
         'n_jobs': 10,
         'eval_metric': 'auc'}
    print('Choosing parameters:', p)
    scores = []
    sizes = []
    # gts = GroupTimeSeriesSplit()
    gts = PurgedGroupTimeSeriesSplit(n_splits=5, group_gap=31)
    for i, (tr_idx, val_idx) in enumerate(gts.split(data, groups=date)):
        sizes.append(len(tr_idx))
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
    print(scores)
    avg_score = weighted_mean(scores, sizes)
    print('Avg Score:', avg_score)
    return avg_score


def loptimize(trial):
    p = {'learning_rate': trial.suggest_uniform('learning_rate', 1e-4, 1e-1),
         'max_leaves': trial.suggest_int('max_leaves', 5, 100),
         'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.3, 0.99),
         'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
         'feature_fraction': trial.suggest_uniform('feature_fraction', 0.3, 0.99),
         'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 1000),
         'lambda_l1': trial.suggest_uniform('lambda_l1', 0.005, 0.05),
         'lambda_l2': trial.suggest_uniform('lambda_l2', 0.005, 0.05),
         'boosting': trial.suggest_categorical('boosting', ['gbdt', 'goss', 'rf']),
         'objective': 'binary',
         'verbose': 1,
         'n_jobs': 10,
         'metric': 'auc'}
    if p['boosting'] == 'goss':
        p['bagging_freq'] = 0
        p['bagging_fraction'] = 1.0
    scores = []
    sizes = []
    # gts = GroupTimeSeriesSplit()
    gts = PurgedGroupTimeSeriesSplit(n_splits=5, group_gap=31)
    for i, (tr_idx, val_idx) in enumerate(gts.split(data, groups=date)):
        sizes.append(len(tr_idx))
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
    print(scores)
    avg_score = weighted_mean(scores, sizes)
    print('Avg Score:', avg_score)
    return avg_score


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
"""
api_token = read_api_token()
neptune.init(api_token=api_token,
             project_qualified_name='jamesmccarthy65/JSMP')
data = load_data('data/', mode='train', overide='filtered_train.csv')
data, target, features, date = preprocess_data(data)
print('creating XGBoost Trials')
xgb_exp = neptune.create_experiment('XGBoost_HPO')
xgb_neptune_callback = opt_utils.NeptuneCallback(experiment=xgb_exp)
study = optuna.create_study(direction='maximize')
study.optimize(optimize, n_trials=500, callbacks=[xgb_neptune_callback])
joblib.dump(study, f'HPO/xgb_hpo_{str(datetime.datetime.now().date())}.pkl')
print('Creating LightGBM Trials')
lgb_exp = neptune.create_experiment('LGBM_HPO')
lgbm_neptune_callback = opt_utils.NeptuneCallback(experiment=lgb_exp)
study = optuna.create_study(direction='maximize')
study.optimize(loptimize, n_trials=500, callbacks=[lgbm_neptune_callback])
joblib.dump(study, f'HPO/lgb_hpo_{str(datetime.datetime.now().date())}.pkl')
