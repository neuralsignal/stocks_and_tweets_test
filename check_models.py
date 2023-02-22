# %%
import pickle

with open('enewsentiment_grid_model.pkl', 'rb') as f:
    model = pickle.load(f)
# %%
model
# %%
dir(model)
# %%
lemodel = model.best_estimator_['model']

# %%
lemodel.estimator.n_estimators
# %%
lemodel.estimator.learning_rate
# %%
model.best_score_
# %%

# %%
model.best_estimator_['pca']

# %%
lemodel
# %%
