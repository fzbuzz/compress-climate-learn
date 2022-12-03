from typing import Any

import torch
from pytorch_lightning import LightningModule
from torchvision.transforms import transforms

from compressai.losses import RateDistortionLoss

from .utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from .utils.metrics import mse, rmse, pearson, mean_bias


class CompressLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: str = 'adam',
        lr: float = 0.001,
        weight_decay: float = 0.005,
        warmup_epochs: int = 5,
        max_epochs: int = 30,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 1e-8,
    ):

        super().__init__()
        self.rd_loss = RateDistortionLoss()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        if optimizer == 'adam':
            self.optim_cls = torch.optim.Adam
        elif optimizer == 'adamw':
            self.optim_cls = torch.optim.AdamW
        else:
            raise NotImplementedError('Only support Adam and AdamW')

    def set_denormalization(self, mean, std):
        self.denormalization = transforms.Normalize(mean, std)

    def set_lat_lon(self, lat, lon):
        self.lat = lat
        self.lon = lon

    def set_pred_range(self, r):
        self.pred_range = r
        
    def set_train_climatology(self, clim):
        self.train_clim = clim

    def set_val_climatology(self, clim):
        self.val_clim = clim

    def set_test_climatology(self, clim):
        self.test_clim = clim

    def forward(self, x):
        with torch.no_grad():
            return self.net(x)['x_hat']

    def training_step(self, batch: Any, batch_idx: int):
        x, y, _, _ = batch
        ret = self.net.forward(x)
        loss_dict = self.rd_loss(ret, y)
        loss = loss_dict['loss']
        for var in loss_dict.keys():
            self.log(
                "train/" + var,
                loss_dict[var],
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                batch_size = len(x)
            )
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        x, y, variables, out_variables = batch

        ret = self.net.forward(x)
        loss_dict = self.rd_loss(ret, y)
        loss_dict['pearson_loss'] = pearson(x,y, variables, transform=self.denormalization)
        loss_dict['mean_bias'] = mean_bias(x,y,variables, transform=self.denormalization)

        for var in loss_dict.keys():
            self.log(
                "val/" + var,
                loss_dict[var],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                batch_size = len(x)
            )
        return loss_dict

    def test_step(self, batch: Any, batch_idx: int):
        x, y, variables, _ = batch

        ret = self.net.forward(x)
        loss_dict = self.rd_loss(ret, y)
        loss_dict['pearson_loss'] = pearson(x,y, variables, transform=self.denormalization)
        loss_dict['mean_bias'] = mean_bias(x,y,variables, transform=self.denormalization)

        for var in loss_dict.keys():
            self.log(
                "test/" + var,
                loss_dict[var],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size = len(x)
            )
        return loss_dict


    def configure_optimizers(self):
        decay = []
        no_decay = []
        for name, m in self.named_parameters():
            if "pos_embed" in name:
                no_decay.append(m)
            else:
                decay.append(m)

        optimizer = self.optim_cls(
            [
                {
                    "params": decay,
                    "lr": self.hparams.lr,
                    "weight_decay": self.hparams.weight_decay,
                },
                {"params": no_decay, "lr": self.hparams.lr, "weight_decay": 0},
            ]
        )

        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            self.hparams.warmup_epochs,
            self.hparams.max_epochs,
            self.hparams.warmup_start_lr,
            self.hparams.eta_min,
        )

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
