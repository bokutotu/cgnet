import os
from pathlib import Path
from glob import glob
from urllib.parse import urlparse
from collections import OrderedDict
import argparse

import torch
from omegaconf import OmegaConf
import mlflow

# from src.setup import setup_model
from src.experiment import Experiment


def get_artifact(run_id):
    uri = mlflow.get_run(run_id).info.artifact_uri
    path = urlparse(uri).path
    local_path = "/".join(path.split("/")[0:-1])
    return local_path


def load_from_run_id(run_id):
    artifacts_path = get_artifact(run_id)
    config_path = os.path.join(artifacts_path, "artifacts/config.yaml")
    model_path = os.path.join(artifacts_path, "artifacts/model.pth")
    config = OmegaConf.load(config_path)
    state_dict = torch.load(model_path)
    print(state_dict)
    model = setup_model(config)
    model.load_state_dict(state_dict)
    return model


def main(args):
    model = load_from_run_id(args.run_id)
    print(model)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, help="experiment id of mlflow")
    args = parser.parse_args()
    main(args)
