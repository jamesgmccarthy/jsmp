# %%
import torch
import torch.nn as nn
import utils
from pytorch_lightning import Callback
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pytorch_lightning as pl


# %%


class AutoEncoder(pl.LightningModule):
    def __init__(self, input_size, output_size, params=None,
                 model_path='models/ae.pth', fold=None):
        super(AutoEncoder, self).__init__()
        self.dim_1 = params['dim_1']
        self.dim_2 = params['dim_2']
        self.dim_3 = params['dim_3']
        self.dim_4 = params['dim_4']
        self.hidden = params['hidden']
        self.dropout_prob = params['dropout']
        self.lr = params['lr']
        self.activation = params['activation']
        self.input_size = input_size
        self.output_size = output_size
        self.aeloss = nn.MSELoss()
        self.loss = nn.BCEWithLogitsLoss()
        self.weight_decay = params['weight_decay']
        self.label_smoothing = params['label_smoothing']
        self.amsgrad = params['amsgrad']
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, self.dim_1, bias=False),
            nn.BatchNorm1d(self.dim_1),
            self.activation(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(self.dim_1, self.dim_2, bias=False),
            nn.BatchNorm1d(self.dim_2),
            self.activation(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(self.dim_2, self.dim_3, bias=False),
            nn.BatchNorm1d(self.dim_3),
            self.activation(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(self.dim_3, self.dim_4, bias=False),
            nn.BatchNorm1d(self.dim_4),
            self.activation(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(self.dim_4, self.hidden, bias=False),
            nn.BatchNorm1d(self.hidden),
            self.activation(),
            nn.Dropout(p=self.dropout_prob)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden, self.dim_4, bias=False),
            nn.BatchNorm1d(self.dim_4),
            self.activation(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(self.dim_4, self.dim_3, bias=False),
            nn.BatchNorm1d(self.dim_3),
            self.activation(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(self.dim_3, self.dim_2, bias=False),
            nn.BatchNorm1d(self.dim_2),
            self.activation(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(self.dim_2, self.dim_1, bias=False),
            nn.BatchNorm1d(self.dim_1),
            self.activation(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(self.dim_1, self.input_size, bias=False),
            nn.BatchNorm1d(self.input_size)
        )

    def forward(self, x):
        z = self.encoder(x)
        return z

    def training_step(self, batch, batch_idx):
        x, y = batch['data'], batch['target']
        y = y.view(-1)
        x = x.view(x.size(1), -1)
        z = self(x)
        x_hat = self.decoder(z)
        loss = self.aeloss(x_hat, x)
        self.log('t_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, amsgrad=self.amsgrad)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.1, min_lr=1e-7, eps=1e-8)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 't_loss'}


def train_autoencoder():
    data = utils.load_data(root_dir='./data/', mode='train')
    data, target, features, date = utils.preprocess_data(data, nn=True)
    dataset = utils.FinData(data=data, target=target, date=date)
    p = {'batch_size': 4597,
         'dim_1': 231,
         'dim_2': 851,
         'dim_3': 777,
         'dim_4': 192,
         'hidden': 50,
         'dropout': 0.017122456592972537,
         'lr': 0.0013131268366473552,
         'activation': nn.GELU,
         'label_smoothing': 0.09401544509474698,
         'weight_decay': 0.005078413740277699,
         'amsgrad': True}
    train_idx = [i for i in range(len(data))]
    val_idx = [i for i in range(10000)]
    dataloaders = utils.create_dataloaders(dataset=dataset,
                                           indexes={
                                               'train': train_idx, 'val': val_idx},
                                           batch_size=p['batch_size'])

    checkpoint_callback = ModelCheckpoint(
        dirpath='logs', monitor='t_loss', mode='min', save_top_k=1, period=10)
    input_size = data.shape[-1]
    output_size = 1
    model = AutoEncoder(input_size=input_size,
                        output_size=output_size, params=p)
    es = EarlyStopping(monitor='t_loss', patience=10,
                       min_delta=0.0005, mode='min')
    trainer = pl.Trainer(max_epochs=500, gpus=1, callbacks=[checkpoint_callback, es],
                         precision=16)
    trainer.fit(model, train_dataloader=dataloaders['train'])


def main():
    train_autoencoder()

    if __name__ == '__main__':
        main()
