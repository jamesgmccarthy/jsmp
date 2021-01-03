# %%
import torch
import torch.nn as nn


# %%


class AutoEncoder(nn.Module):
    def __init__(self, input_size, output_size, dims, drop, batch_size, learning_rate=0.05, early_stopping=10,
                 model_path='models/ae.pth', fold=None):
        self.inp_size = input_size
        self.out_size = output_size
        self.dims = dims
        self.dp = drop
        self.bs = batch_size
        self.lr = learning_rate
        self.es = early_stopping
        self.mp = self.create_model_file(model_path, fold)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self
