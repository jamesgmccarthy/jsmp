import pytorch_lightning as pl


class Autoencoder(pl.LightningModule):
    def __init__(self):
        super(Autoencoder, self).__init__()
        