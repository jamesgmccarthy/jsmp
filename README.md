# Kaggle Competition - Jane Street Market Prediction

Repository for my submission to JSMP competition. The competition has finished taking submissions. 
First run of private leaderboard. Currently in top 8%.

Focus has been about building a functioning pipeline (i.e. data processing to prediction) that I can reimplement in later projects.
Hyperparameter search has been added to optimize models performance on base dataset. Optuna is used in conjunction with Neptune.ai to monitor experiments.

Models used are XGBoost, LightGBM, simple MLP and ResNet models in PyTorch lightning. The NN's proved more difficult to implement properly in pytorch lightning than I
originally hoped as their documentation is somewhat lacking. However, once I got it working it proved very useful as it handles a lot of boilerplate code and makes 
it very easy to reuse code and change models while keeping it readable.

## Solution

I followed a relatively simple approach to this problem as I wanted to focus on creating a pipeline I could re-use.

### Data Processing and Cross-Validation
- 5 fold Purged Time Series Cross Validation using 5-day gap - Gap used to prevent possible leakage 
- First 85 days removed - Findings showed target vairable had high variance in first 85 days. Believed whatever model used by Jane Street to create repsonse variable was performing poorly and changed afterwards.
- Missing values were filled with mean of training datasets
- Target variable was all 5 response variables for a binary multilabel classification target
- Didn't mess with threshold of 0.5, felt like didn't have any meaningful reason to.
- Mean of all 
