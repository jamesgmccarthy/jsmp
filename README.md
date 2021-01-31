# Kaggle Competition - Jane Street Market Prediction

Repository for my submission to JSMP competition. Currently a work in progress

# Notes on current progress
So far the focus has been building a functioning pipeline (i.e. Data processing to prediction) that I can reimplement in later projects.
Hyperparameter search has been added to optimize models performance on base dataset. Optuna is used in conjunction with Neptune.ai to monitor experiments.

Current models used are XGBoost, LightGBM and simple MLP in pytorch lightning. The MLP proved more difficult to implement properly in pytorch lightning than I
originally hoped as their documentation is somewhat lacking. However, once I got it working it proved very useful as it handles a lot of boilerplate code and makes 
it very easy to reuse code and change models while keeping it readable.

None of the models are currently doing very well and I'm stuck just show of the top 50%. I believe there's a problem with the full training loop of the model and 
that its over fitting to the training data, despite implementing early stopping, dropout, batch norm etc.
Note:
  Exploration still to come, to add features and adjust models used, while hoping little to no affect to pipeline.
  
  
