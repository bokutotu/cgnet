import sys
sys.path.append("cgnet")

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer

from hydra.utils import instantiate
from omegaconf import DictConfig
import numpy as np

from cgnet.network.nnet import CGnet, HarmonicLayer, ForceLoss, ZscoreLayer
from cgnet.feature import GeometryFeature, GeometryStatistics

from src.model import MLP


class Experiment(pl.LightningModule):
    def __init__(self, config: DictConfig) -> None:
        super(Experiment, self).__init__()
        self.config: DictConfig = config
        logger = instantiate(config.logger)
        self.trainer = instantiate(
            config.trainer,
            logger=logger,
            callbacks=[
                LearningRateMonitor(logging_interval="step"),
            ],
        )

        coordinates = np.load(config.coordinates)
        forces = np.load(config.forces)

        self.stats = GeometryStatistics(
                coordinates, backbone_inds='all', get_all_distances=True, 
                get_backbone_angles=True, get_backbone_dihedrals=True
        )

        zscores, _ = self.stats.get_zscore_array()
        all_stats, _ = stats.get_prior_statistics(as_list=True)
        nnet = MLP(len(all_stats))
        layers = [ZscoreLayer(zscores)]
        layers += [nnet]
        feature_layer = GeometryFeature(feature_tuples=self.stats.feature_tuples)

        # prior layer
        bond_list, bond_keys = stats.get_prior_statistics(features='Bonds', as_list=True)
        bond_indices = stats.return_indices('Bonds')
        angle_list, angle_keys = stats.get_prior_statistics(features='Angles', as_list=True)
        angle_indices = stats.return_indices('Angles')
        priors  = [HarmonicLayer(bond_indices, bond_list)]
        priors += [HarmonicLayer(angle_indices, angle_list)]

        self.model = CGnet(layers, ForceLoss, feature=feature_layer, priors=priors)


    def configure_optimizers(self):
        params = self.model.parameters()
        optimizer: Optimizer = instantiate(self.config.optimizer, params=params)
        scheduler = instantiate(self.config.scheduler, optimizer=optimizer)
        return [optimizer], [scheduler]

    def loss_fn(self, recon_x: Tensor, x: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
        BCE = F.binary_cross_entropy_with_logits(
            recon_x, x.view(-1, 784), reduction="sum"
        )
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def training_step(self, batch: Tensor, batch_idx: int):
        recon_batch, mu, logvar = self.model(batch)
        loss = self.loss_fn(recon_batch, batch, mu, logvar)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int):
        recon_batch, mu, logvar = self.model(batch)
        loss = self.loss_fn(recon_batch, batch, mu, logvar)
        self.log("val_loss", loss)
        return loss

    # train your model
    def fit(self):
        self.trainer.fit(self, self.data_module)
        self.logger.log_hyperparams(
            {
                "batch_size": self.config.batch_size,
                "lr": self.config.lr,
            }
        )
        self.log_artifact(".hydra/config.yaml")
        self.log_artifact(".hydra/hydra.yaml")
        self.log_artifact(".hydra/overrides.yaml")
        self.log_artifact("main.log")

    # run your whole experiments
    def run(self):
        self.fit()

    def log_artifact(self, artifact_path: str):
        self.logger.experiment.log_artifact(self.logger.run_id, artifact_path)
