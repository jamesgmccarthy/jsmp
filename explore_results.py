# %%
import joblib
import optuna

# %%
hpo: optuna.Study = joblib.load('HPO/nn_hpo_2021-01-21.pkl')

# %%
hpo.best_params
# %%
