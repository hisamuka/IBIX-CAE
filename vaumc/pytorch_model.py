"""
PyTorch model handling
Author: Mathijs de Boer
Date Created: 2021-04-28
"""

from pathlib import Path
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from pl_bolts.models.gans.pix2pix.pix2pix_module import Pix2Pix
from pytorch_msssim import ssim, ms_ssim
from skimage.util import view_as_windows
from torch.nn import functional as f


def __normalize(x):
    x_min = torch.min(x)
    x_max = torch.max(x)

    return (x - x_min) / (x_max - x_min)


def weighted_l1_loss(y_hat, y, weight):
    importance = (y - torch.min(y)) / (torch.max(y) - torch.min(y))
    importance *= weight

    error = (y_hat - y) * importance
    absolute = torch.abs(error)
    mean = torch.mean(absolute)

    return mean


def ssim_loss(y_hat, y):
    # This SSIM implementation requires the input images to be in rage [0.0, 1.0]
    norm_y_hat = __normalize(y_hat)
    norm_y = __normalize(y)
    ssim_val = ssim(norm_y_hat, norm_y, data_range=1, nonnegative_ssim=True)

    # As SSIM is an accuracy metric, a value of 1.0 means a perfect match
    # Loss is generally a minimization problem, so we invert the direction
    return 1.0 - ssim_val


def l1_msssim_loss(y_hat, y, alpha=0.83):
    """
    From Zhao et al. (2015) https://arxiv.org/abs/1511.08861
    """
    norm_y_hat = __normalize(y_hat)
    norm_y = __normalize(y)
    ms_ssim_val = 1.0 - ms_ssim(norm_y_hat, norm_y, data_range=1)
    l1_loss_val = f.l1_loss(y_hat, y)

    return alpha * ms_ssim_val + (1 - alpha) * l1_loss_val


class Weighted_L1(pl.LightningModule):
    def __init__(self, weight):
        super(Weighted_L1, self).__init__()
        self.weight = weight

    def forward(self, pred, y):
        return weighted_l1_loss(pred, y, self.weight)


class L1MSSSIM(pl.LightningModule):
    def __init__(self):
        super(L1MSSSIM, self).__init__()

    def forward(self, pred, y):
        return l1_msssim_loss(pred, y)


class SSIM(pl.LightningModule):
    def __init__(self):
        super(SSIM, self).__init__()

    def forward(self, pred, y):
        return ssim_loss(pred, y)


class P2PNet(Pix2Pix):
    """
    Wrapper around the Pix2Pix to add validation and testing steps
    """

    def __init__(
            self,
            input_channels: int = 3,
            output_channels: int = 1,
            learning_rate: float = 0.0002,
            lambda_recon: float = 200,
            b1: float = 0.5,
            b2: float = 0.999,
            loss_function: str = "l1"
    ):
        super(P2PNet, self).__init__(
            in_channels=input_channels,
            out_channels=output_channels,
            learning_rate=learning_rate,
            lambda_recon=lambda_recon
        )

        self.lf = loss_function

        if self.lf == "l1":
            pass
        elif self.lf == "l1_msssim":
            self.recon_criterion = L1MSSSIM()
        elif self.lf == "ssim":
            self.recon_criterion = SSIM()
        elif self.lf == "weighted_l1":
            self.recon_criterion = Weighted_L1(weight=1.25)
        else:
            raise ValueError(f"Validation Loss \"{self.lf}\" is not a valid loss function")

        self.save_hyperparameters()

    def forward(self, x):
        return self.gen(x)

    def validation_step(self, batch, batch_idx):
        y, x = batch

        pred = self(x)
        l1 = f.l1_loss(pred, y)
        ssim = ssim_loss(pred, y)
        l1_msssim = l1_msssim_loss(pred, y)
        weighted_l1 = weighted_l1_loss(pred, y, 1.25)

        if self.lf == "l1_msssim":
            loss = l1_msssim
        elif self.lf == "l1":
            loss = l1
        elif self.lf == "ssim":
            loss = ssim
        elif self.lf == "weighted_l1":
            loss = weighted_l1
        else:
            raise ValueError(f"Validation Loss \"{self.lf}\" is not a valid loss function")

        # log sampled images
        sample_imgs = torch.cat(
            (
                x[:4],
                torch.cat((y[:4], y[:4], y[:4]), dim=1),
                torch.cat((pred[:4], pred[:4], pred[:4]), dim=1)
            ),
            dim=0
        )
        grid_gen = torchvision.utils.make_grid(sample_imgs, nrow=4, normalize=True, scale_each=True)
        self.logger.experiment.add_image('val_results', grid_gen, self.current_epoch)

        self.log("ptl/val_loss", loss, prog_bar=False, logger=True)
        self.log("ptl/val_ssim", 1 - ssim, prog_bar=False, logger=True)
        self.log("ptl/val_l1_msssim", l1_msssim, prog_bar=False, logger=True)
        self.log("ptl/val_l1", l1, prog_bar=False, logger=True)
        self.log("ptl/val_weighted_l1", weighted_l1, prog_bar=False, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        y, x = batch

        pred = self(x)
        l1 = f.l1_loss(pred, y)
        ssim = ssim_loss(pred, y)
        l1_msssim = l1_msssim_loss(pred, y)
        weighted_l1 = weighted_l1_loss(pred, y, 1.25)

        if self.lf == "l1_msssim":
            loss = l1_msssim
        elif self.lf == "l1":
            loss = l1
        elif self.lf == "ssim":
            loss = ssim
        elif self.lf == "weighted_l1":
            loss = weighted_l1
        else:
            raise ValueError(f"Validation Loss \"{self.lf}\" is not a valid loss function")

        return {
            "test_loss": loss,
            "test_ssim": ssim,
            "test_l1_msssim": l1_msssim,
            "test_l1": l1,
            "test_weighted_l1": weighted_l1
        }

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_ssim = torch.stack([x["test_ssim"] for x in outputs]).mean()
        avg_l1_msssim = torch.stack([x["test_l1_msssim"] for x in outputs]).mean()
        avg_l1 = torch.stack([x["test_l1"] for x in outputs]).mean()
        avg_weighted_l1 = torch.stack([x["test_weighted_l1"] for x in outputs]).mean()

        self.log("ptl/test_loss", avg_loss, prog_bar=False, logger=True)
        self.log("ptl/test_ssim", 1 - avg_ssim, prog_bar=False, logger=True)
        self.log("ptl/test_l1_msssim", avg_l1_msssim, prog_bar=False, logger=True)
        self.log("ptl/test_l1", avg_l1, prog_bar=False, logger=True)
        self.log("ptl/test_weighted_l1", avg_weighted_l1, prog_bar=False, logger=True)

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(b1, b2))
        disc_opt = torch.optim.Adam(self.patch_gan.parameters(), lr=lr, betas=(b1, b2))

        # We do also reduce lr as we stop improving
        scheduler_disc = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                disc_opt, factor=0.5, patience=50, verbose=True),
            "monitor": "ptl/val_loss"
        }
        scheduler_gen = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                gen_opt, factor=0.5, patience=50, verbose=True),
            "monitor": "ptl/val_loss"
        }

        return [disc_opt, gen_opt], [scheduler_disc, scheduler_gen]


def load_model(path: Union[str, Path]) -> torch.nn.Module:
    has_cuda = torch.cuda.is_available()
    print(f"Has_CUDA = {has_cuda}")
    model = P2PNet()
    try:
        model.load_state_dict(torch.load(path))
        if has_cuda:
            model.to("cuda")
    except Exception as err:
        print(err)
        raise err
    model.eval()
    return model


def reconstruct_image(image: np.ndarray, model: torch.nn.Module) -> np.ndarray:
    # Pix2Pix assumes 256x256 input, so we do too
    input_size = 256
    step_size = (image.shape[0] - input_size) // 1
    if step_size < 1:
        step_size = 1
    has_cuda = torch.cuda.is_available()

    prediction = np.zeros((image.shape[0], image.shape[1]))
    # We combine several predictions, so we need to average out overlapping areas
    divider = np.zeros_like(prediction)

    # `view_as_windows` allows us to edit things in the source array, too
    image_batch = view_as_windows(image, (input_size, input_size, 3), step=step_size)
    prediction_batch = view_as_windows(prediction, (input_size, input_size), step=step_size)
    divider_batch = view_as_windows(divider, (input_size, input_size), step=step_size)

    # Do some reshaping to turn the batch shape into (n, c, x, y)
    image_batch = np.squeeze(image_batch)
    if image_batch.ndim > 3:
        image_batch = image_batch.reshape((image_batch.shape[0] * image_batch.shape[1],) + image_batch.shape[2:])
        image_batch = np.moveaxis(image_batch, -1, 1)
    else:
        image_batch = np.moveaxis(image_batch, -1, 0)
        image_batch = np.expand_dims(image_batch, 0)

    prediction_batch = np.squeeze(prediction_batch)
    divider_batch = np.squeeze(divider_batch)

    try:
        t = torch.from_numpy(image_batch).to("cuda" if has_cuda else "cpu")
        p = model(t)
    except Exception as err:
        print(err)
        raise err

    if has_cuda:
        p = np.squeeze(p.cpu().detach().numpy())
    else:
        p = np.squeeze(p.detach().numpy())
    p = p.reshape(prediction_batch.shape)

    for i in range(prediction_batch.shape[0]):
        for j in range(prediction_batch.shape[1]):
            # Any changes made to prediction_batch also end up in prediction
            prediction_batch[i, j] += p[i, j]
            divider_batch[i, j] += 1

    prediction = np.divide(prediction, divider, out=np.zeros_like(prediction), where=divider != 0)
    return prediction


def reconstruct_image_set(image_set: np.ndarray, model: torch.nn.Module) -> np.ndarray:
    # Assume a [batch, c, h, w] order
    has_cuda = torch.cuda.is_available()
    image_set = np.moveaxis(image_set, -1, 1)

    try:
        t = torch.from_numpy(image_set).to("cuda" if has_cuda else "cpu")
        p = model(t)
    except Exception as err:
        print(err)
        raise err
    if has_cuda:
        p = np.squeeze(p.cpu().detach().numpy())
    else:
        p = np.squeeze(p.detach().numpy())

    prediction = p
    print(prediction.shape)

    return prediction
