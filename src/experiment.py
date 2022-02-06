import sys
sys.path.append("cgnet")

import os
from urllib.parse import urlparse

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
from src.data_module import DataModule


def prepare_model(coordinates: np.array, is_bond_prior: bool, is_angle_prior: bool, 
        is_dihedral_prior: bool, net_config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    stats = GeometryStatistics(
            coordinates[::100], backbone_inds='all', get_all_distances=True, 
            get_backbone_angles=True, get_backbone_dihedrals=True
    )

    zscores, _ = stats.get_zscore_array()
    all_stats, _ = stats.get_prior_statistics(as_list=True)
    nnet = instantiate(net_config, input_size=len(all_stats))
    layers = [ZscoreLayer(zscores)]
    layers += [nnet]
    feature_layer = GeometryFeature(feature_tuples=stats.feature_tuples, device=device)

    # prior layer
    priors = []
    if is_bond_prior:
        bond_list, bond_keys = stats.get_prior_statistics(features='Bonds', as_list=True)
        bond_indices = stats.return_indices('Bonds')
        priors += [HarmonicLayer(bond_indices, bond_list)]
    if is_angle_prior:
        angle_list, angle_keys = stats.get_prior_statistics(features='Angles', as_list=True)
        angle_indices = stats.return_indices('Angles')
        priors += [HarmonicLayer(angle_indices, angle_list)]
    if is_dihedral_prior:
        dihedral_sin_list, _ = stats.get_prior_statistics(features='Dihedral_sines', as_list=True)
        dihedral_cos_list, _ = stats.get_prior_statistics(features='Dihedral_cosines', as_list=True)
        dihedral_cos_indices = stats.return_indices("Dihedral_cosines")
        dihedral_sin_indices = stats.return_indices("Dihedral_sines")
        priors += [HarmonicLayer(dihedral_sin_indices, dihedral_sin_list)]
        priors += [HarmonicLayer(dihedral_cos_indices, dihedral_cos_list)]


    # self.model = CGnet(layers, ForceLoss(), feature=feature_layer, priors=priors)
    model = CGnet(layers, torch.nn.MSELoss(), feature=feature_layer, priors=priors)
    return model


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

        self.model = prepare_model(coordinates=coordinates, 
                is_bond_prior=config.is_bond_prior, is_angle_prior=config.is_angle_prior,
                is_dihedral_prior=config.is_dihedral_prior, net_config=config.model)

        if "MLP" in config.model._target_.split("."):
            mode = "mlp"
        else:
            mode = "time"
        self.data_module = DataModule(
                batch_size=config.batch_size, train_test_rate=config.train_test_rate, 
                coordinates=coordinates, forces=forces, mode=mode, scale=config.scale,
                feature_length=config.feature_length
        )

        self.best_model_state_dict = self.model.state_dict()
        self.val_loss = 1e20

    def configure_optimizers(self):
        params = self.model.parameters()
        optimizer: Optimizer = instantiate(self.config.optimizer, params=params)
        scheduler = instantiate(self.config.scheduler, optimizer=optimizer)
        return [optimizer], [scheduler]

    def loss_fn(self, predicted_force: Tensor, force: Tensor) -> Tensor:
        batch_loss = self.model.criterion(predicted_force, force) 
        return batch_loss

    def _step(self, coords: Tensor, force: Tensor) -> Tensor:
        potential, predicted_force = self.model.forward(coords)
        return potential, predicted_force

    def set_requires_grad(self, x: Tensor, y: Tensor) -> (Tensor, Tensor):
        return x.requires_grad_(True), y.requires_grad_(True)

    def training_step(self, batch: Tensor, batch_idx: int):
        coords, force, _ = batch
        coords, force = self.set_requires_grad(coords, force)
        potential, predicted_force = self._step(coords, force)
        loss = self.loss_fn(predicted_force, force)
        # self.log("train_loss", loss)
        return loss

    def training_epoch_end(self, loss):
        loss = np.array([float(item["loss"].detach().cpu()) for item in loss])
        loss_avg = loss.mean()
        self.log("train_loss", loss_avg)

    @torch.enable_grad()
    def validation_step(self, batch: Tensor, batch_idx: int):
        coords, force, _ = batch
        coords, force = self.set_requires_grad(coords, force)
        potential, predicted_force = self._step(coords, force)
        loss = self.loss_fn(predicted_force, force)
        # self.log("val_loss", loss)
        return loss

    def validation_epoch_end(self, loss):
        loss = np.array([float(i.detach().cpu()) for i in loss])
        loss_avg = loss.mean()

        if loss_avg <= self.val_loss:
            self.val_loss = loss_avg
            self.best_model_state_dict = self.model.state_dict()
        self.log("validation_loss", loss_avg)

    @torch.enable_grad()
    def test_step(self, batch: Tensor, batch_idx: int):
        coords, force, _ = batch
        coords, force = self.set_requires_grad(coords, force)
        potential, predicted_force = self._step(coords, force)
        loss = self.loss_fn(predicted_force, force)
        # self.log("test_loss", loss)
        return loss

    def test_epoch_end(self, loss):
        loss = np.array([float(i.detach().cpu()) for i in loss])
        loss_avg = loss.mean()

        self.log("test_loss", loss_avg)

    def save(self):
        artifact_path = urlparse(self.logger._tracking_uri).path
        self.artifact_path = os.path.join(
            artifact_path, self.logger.experiment_id, self.logger.run_id, "artifacts")
        torch.save(self.best_model_state_dict, self.artifact_path + "/model.pth")

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
        self.trainer.test()
        self.save()

    def log_artifact(self, artifact_path: str):
        self.logger.experiment.log_artifact(self.logger.run_id, artifact_path)
