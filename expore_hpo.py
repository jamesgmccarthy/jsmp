# %%
import joblib
import optuna

# %%
lgb = joblib.load('lgb_hpo.pkl')
xgb = joblib.load('xgb_hpo.pkl')

# %%
optuna.visualization.plot_param_importances(lgb)
# %%
optuna.visualization.plot_param_importances(xgb)
# %%
optuna.visualization.plot_parallel_coordinate(lgb)
# %%
optuna.visualization.plot_parallel_coordinate(xgb)
# %%
optuna.visualization.plot_slice(lgb)
# %%
lgb.best_params
# %%
lgb.best_trial.params
# %%
