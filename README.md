# Kaggle Competition - Jane Street Market Prediction

Repository for my submission to JSMP competition. The competition has finished taking submissions. 
First run of private leaderboard. Currently in top 8%.

Focus has been about building a functioning pipeline (i.e. data processing to prediction) that I can reimplement in later projects.
Hyperparameter search has been added to optimize models performance on base dataset. Optuna is used in conjunction with Neptune.ai to monitor experiments.

Current models used are XGBoost, LightGBM, simple MLP and ResNet models in PyTorch lightning. The NN's proved more difficult to implement properly in pytorch lightning than I
originally hoped as their documentation is somewhat lacking. However, once I got it working it proved very useful as it handles a lot of boilerplate code and makes 
it very easy to reuse code and change models while keeping it readable.


#
