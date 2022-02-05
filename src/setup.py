import multiprocessing

import numpy as np
import torch
from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler
from torch import optim

from hydra.utils import instantiate

from src.model import CGnet
from src.dataset import MLPDataset, LSTMDataset
from src.statics import get_statics
from src.layers.cmap import CMAP, prepare_cmap_force_grad


def setup_model(cfg):
