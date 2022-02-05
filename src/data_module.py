import sys
sys.path.append("cgnet")

import os
from typing import Optional

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from cgnet.feature.dataset import MoleculeDataset, MoleculeTimeSeriseDataset


class DataModule(pl.LightningDataModule):
    def __init__(
            self, batch_size: int, train_test_rate: float, coordinates: np.array, 
            forces: np.array, mode: str, scale: float, feature_length=None
        ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.mode = mode
        self.feature_length = feature_length
        coordinates = coordinates.astype(np.float32)
        forces = forces.astype(np.float32) * scale
        print("force mean abs", np.mean(np.abs(forces)))

        len_coord = len(coordinates)

        train_last_idx = int(pow(train_test_rate, 2) * len_coord)
        val_last_idx = int(train_test_rate * len_coord)

        self.train_coord = coordinates[0:train_last_idx]
        self.train_force = forces[0: train_last_idx]
        self.val_coord = coordinates[train_last_idx: val_last_idx]
        self.val_force = forces[train_last_idx: val_last_idx]
        self.test_coord = coordinates[val_last_idx:-1]
        self.test_force = forces[val_last_idx: -1]

        self.train = None
        self.val = None
        self.test = None

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        # make assignments here (val/train/test split)
        # called on every GPUs
        if self.mode == "mlp":
            self.train = MoleculeDataset(coordinates=self.train_coord, forces=self.train_force)
            self.val = MoleculeDataset(coordinates=self.val_coord, forces=self.val_force)
            self.test = MoleculeDataset(coordinates=self.test_coord, forces=self.test_force)
        else:
            self.train = MoleculeTimeSeriseDataset(
                    coordinates=self.train_coord, forces=self.train_force, 
                    feature_length=self.feature_length)
            self.val = MoleculeTimeSeriseDataset(
                    coordinates=self.val_coord, forces=self.val_force, 
                    feature_length=self.feature_length)
            self.test = MoleculeTimeSeriseDataset(
                    coordinates=self.test_coord, forces=self.test_force, 
                    feature_length=self.feature_length)


    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            # pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            # pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            # pin_memory=True,
        )

    def teardown(self, stage: Optional[str] = None):
        # clean up after fit or test
        # called on every process in DDP
        pass
