# Kaggle Competition - Jane Street Market Prediction

Repository for my submission to JSMP competition. The competition has finished taking submissions. 
First run of private leaderboard. Currently in top 12%.

Focus has been about building a functioning pipeline (i.e. data processing to prediction) that I can reimplement in later projects.
Hyperparameter search has been added to optimize models performance on base dataset. Optuna is used in conjunction with Neptune.ai to monitor experiments.

Models used are XGBoost, LightGBM, simple MLP and ResNet models in PyTorch lightning. The NN's proved more difficult to implement properly in pytorch lightning than I
originally hoped as their documentation is somewhat lacking. However, once I got it working it proved very useful as it handles a lot of boilerplate code and makes 
it very easy to reuse code and change models while keeping it readable.

The final notebook can be found here, cleaned slightly since submission but model and training remain unchanged 
https://www.kaggle.com/jamesmccarthy65/jane-street-pytorch-lightning-nn-pgts
## Solution

I followed a relatively simple approach to this problem as I wanted to focus on creating a pipeline I could re-use.

### Data Processing and Cross-Validation
- 5 fold Purged Time Series Cross Validation using 5-day gap used to tune hyperparameters - Gap used to prevent possible leakage 
- First 85 days removed - Findings showed target vairable had high variance in first 85 days. Believed whatever model used by Jane Street to create repsonse variable was performing poorly and changed afterwards.
- Missing values were filled with mean of training datasets
- Target variable was (weight * response >0) for a binary classification target

### Model
5 layer ResNet model written in Pytorch Lightning. Hyperparameters were arrived at using multiple cross-validation trials with Optuna. 
- Layer Dims: [167, 454, 371, 369, 155]
- Activation Function: Leaky Relu, used to avoid the "dying ReLu" problem.
- Dropout: 0.2106 
- Learning Rate: 0.0022

