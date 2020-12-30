# %%
from optuna import visualization as vs
from optuna import study
import joblib

hpo = joblib.load('lgb_hop.pkl')

vs.plot_contour(hpo, params=['learning_rate',
                             'max_leaves'])

# %%

hpo.best_params
# %%
