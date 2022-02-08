import os
from pathlib import Path
from glob import glob
from urllib.parse import urlparse
from collections import OrderedDict
import argparse

import torch
from omegaconf import OmegaConf, DictConfig
import mlflow
import numpy as np

from src.experiment import prepare_model


def setup_model(config: DictConfig):
    coordinates = np.load(config.coordinates)
    model = prepare_model(coordinates=coordinates, net_config=config.model, 
            is_bond_prior=config.is_bond_prior, is_angle_prior=config.is_angle_prior, 
            is_dihedral_prior=config.is_dihedral_prior) 
    return model


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
    model = setup_model(config)
    model.load_state_dict(state_dict)
    return model, config


def load_npy_float32(path: str):
    return np.load(path).astype(np.float32)


def load_init_state(c_path: str, v_path: str, f_path: str):
    return load_npy_float32(c_path), load_npy_float32(v_path), load_npy_float32(f_path)


def verlet_coord(c: np.array, v: np.array, f: np.array):
    """
    c[t+1] = c[t] + v[t] + 0.5 * f[t]
    """
    return c + v + 0.5 * f


def verlet_velocity(v: np.array, f_prev: np.array, f: np.array):
    """
    v[t] = v[t-1] + 0.5(f[t-1] + f[t])
    """
    return v + 0.5 * (f_prev + f)


def velocity_scaling(v: np.array, norm: float):
    velocity_sum_squre = np.sum(np.power(v, 2))
    v = 1.8 * v * norm / velocity_sum_squre
    return v


def init_state_mlp(coordinates: np.array, velocity: np.array, forces: np.array, num_steps: int):
    num_atoms = coordinates.shape[1]
    return coordinates[0:num_steps], velocity[0:num_steps], forces[0:num_steps]


def init_state_lstm(coordinates: np.array, velocity: np.array, forces: np.array, 
        num_steps: int, feature_length: int):
    return coordinates[0:num_steps+feature_length], \
            velocity[0:num_steps+feature_length], \
            forces[0:num_steps+feature_length]


def force_mlp(scale: float, coordinates: np.array, forces: np.array, step: int, model: torch.nn.Module, 
        *args, **kwargs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    coordinates_tensor = torch.tensor(coordinates[step]).to(device)
    coordinates_tensor = torch.unsqueeze(coordinates_tensor, 0).requires_grad_(True)
    _, pred_forces = model(coordinates_tensor)
    forces[step] = pred_forces.detach().cpu().numpy() / scale
    return forces


def update_mlp(coordinates: np.array, velocity: np.array, forces: np.array, step: int, *args, **kwargs):
    velocity[step] = verlet_velocity(velocity[step-1], forces[step-1], forces[step])
    norm = np.sum(np.power(velocity[0], 2))
    velocity[step] = velocity_scaling(velocity[step], norm)
    coordinates[step+1] = verlet_coord(coordinates[step], velocity[step], forces[step])
    return coordinates, velocity


def force_lstm(scale: float, coordinates: np.array, forces: np.array, step: int, 
        feature_length: int, model: torch.nn.Module):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_atoms = coordinates.shape[1]
    coordinates_tensor = torch.tensor(coordinates[step: step+feature_length]).to(device)
    coordinates_tensor = torch.unsqueeze(coordinates_tensor, 0).requires_grad_(True)
    _, pred_forces = model(coordinates_tensor)
    pred_forces = pred_forces[0,-1].detach().cpu().numpy()
    forces[step+feature_length] = pred_forces / scale
    return forces


def update_lstm(
        coordinates: np.array, velocity: np.array, forces: np.array, step: int, feature_length: int):
    velocity[step+feature_length] = \
        verlet_velocity(velocity[step-1+feature_length], forces[step-1+feature_length], forces[step+feature_length])
    norm = np.sum(np.power(velocity[4], 2))
    velocity[step+feature_length] = velocity_scaling(velocity[step+feature_length], norm)
    coordinates[step+1+feature_length] = \
            verlet_coord(coordinates[step+feature_length], velocity[step+feature_length], forces[step+feature_length])
    return coordinates, velocity


def test_func(scale: float, coordinates: np.array, forces: np.array, step: int, *args, **kwargs):
    return forces


def simulate(
        coordinates: np.array, velocity: np.array,
        forces: np.array, num_steps: int, model: torch.nn.Module, 
        config: DictConfig, force_func, update_func, save_name:str, save_dir):
    for step in range(1, num_steps-1):
        forces = force_func(scale=config.scale, coordinates=coordinates, 
                forces=forces, step=step, feature_length=config.feature_length, model=model)
        coordinates, velocity = update_func(coordinates=coordinates, velocity=velocity, 
                forces=forces, step=step, feature_length=config.feature_length)

    dir = Path(save_dir)
    dir.mkdir(parents=True, exist_ok=True)
    np.save(dir / ("c_" + save_name), coordinates)
    np.save(dir / ("f_" + save_name), forces)
    np.save(dir / ("v_" + save_name), velocity)
    


def main(args):
    model, config = load_from_run_id(args.run_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    coordinates, velocity, forces = load_init_state(
            config.coordinates, args.velocity, config.forces)

    for sim_step in range(args.num_sim):
        if "MLP" in config.model._target_.split("."):
            force_func = force_mlp
            trj_coordinates, trj_velocity, trj_forces = init_state_mlp(
                    coordinates[sim_step * args.step_width::], 
                    velocity[sim_step * args.step_width::], 
                    forces[sim_step * args.step_width::], 
                    args.num_steps)
            update_func = update_mlp
        else:
            force_func = force_lstm
            trj_coordinates, trj_velocity, trj_forces = init_state_lstm(
                    coordinates[sim_step * args.step_width::], 
                    velocity[sim_step * args.step_width::], 
                    forces[sim_step * args.step_width::], 
                    args.num_steps, config.feature_length)
            update_func = update_lstm

        if args.mode == "test":
            force_func = test_func
            trj_coordinates, trj_velocity, trj_forces = init_state_mlp(coordinates, velocity, forces, 
                    args.num_steps)
            update_func=update_mlp

        simulate(trj_coordinates, trj_velocity, trj_forces, args.num_steps, model, config, 
                force_func, update_func, save_name=("{}_".format(sim_step) + args.save_name), 
                save_dir=args.save_dir)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, help="experiment id of mlflow")
    parser.add_argument("--num_steps", type=int, help="シミュレーションのステップ数")
    parser.add_argument("--mode", type=str, default="nomal", help="normalは普通に動いて,testのときはtestする")
    parser.add_argument("--velocity", type=str, help="速度のトラジェクトリへのパス")
    parser.add_argument("--save_dir", type=str, help="trj save dir")
    parser.add_argument("--save_name", type=str, help="保存名")
    parser.add_argument("--num_sim", type=int, help="シミュレーションの回数")
    parser.add_argument("--step_width", type=int, help="シミュレーションの初期構造の比較")
    args = parser.parse_args()
    main(args)

