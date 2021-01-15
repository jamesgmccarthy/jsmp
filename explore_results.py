# %%
import joblib
import optuna

# %%
hpo: optuna.Study = joblib.load('HPO/nn_hpo_2021-01-10.pkl')
# %%
hpo.trials[55].params
# %%

# %%
hpo.best_params
