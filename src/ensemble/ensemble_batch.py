import os
import sys
import argparse
from dataclasses import dataclass
from typing import Sequence
from contextlib import contextmanager
import time
from pathlib import Path
from typing import Dict, Mapping, Any, Optional
# import warnings
# warnings.simplefilter('ignore')

import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from l5kit.evaluation import write_pred_csv

sys.path.append("./src")
sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))
from lib.mixture.batch_spherical_gmm import BatchSphericalGMM
from lib.functions.nll import pytorch_neg_multi_log_likelihood_batch


NUM_TEST = 71122
SEED = 0
np.random.seed(SEED)


@dataclass
class EnsembleFlags:
    output_path: str
    weight: Sequence[float]
    sigma: float
    N_sample: int
    file_list: Sequence[str]
    batch_size: int
    device: str
    gmm_kwargs: Mapping[str, Any]
    give_centroids_init: bool
    precisions_init_sigma: Optional[float]


@contextmanager
def add_time(name: str, store: Dict[str, float]):
    t = time.time()
    yield
    if name not in store:
        store[name] = 0.0
    store[name] += time.time() - t


def load_predictions(file_list, weights):
    coords_list, confs_list = [], []
    for path, w in zip(file_list, weights):
        path = Path(path)

        if path.name.endswith(".npz"):
            npz = np.load(str(path))
            confs = npz["confs"]
            coords = npz["coords"]
            n_example, n_modes = confs.shape
            coords = coords.reshape(n_example, n_modes, 50, 2)
            confs = w * confs
        elif path.name.endswith(".csv"):
            df = pd.read_csv(path)
            columns = [f"coord_{xy}{mode}{step}" for mode in range(3) for step in range(50) for xy in ["x", "y"]]
            coords = df.loc[:, columns].to_numpy().reshape(-1, 3, 50, 2)
            confs = w * df.loc[:, ["conf_0", "conf_1", "conf_2"]].to_numpy().reshape(-1, 3)
        else:
            raise ValueError

        coords_list.append(coords)
        confs_list.append(confs)

    coords = np.concatenate(coords_list, axis=1)
    confs = np.concatenate(confs_list, axis=1)
    confs = confs / confs.sum(axis=1, keepdims=True)
    return coords, confs


def load_metadata(file_list):
    path = file_list[0]

    if str(path).endswith(".npz"):
        npz = np.load(path)
        timestamps = npz["timestamps"]
        track_ids = npz["track_ids"]
        targets = npz.get("targets")
        target_availabilities = npz.get("target_availabilities")
        return timestamps, track_ids, targets, target_availabilities
    else:
        df = pd.read_csv(path)
        return df["timestamp"], df["track_id"], None, None


def ensemble_batch_core(flags: EnsembleFlags):
    assert len(flags.weight) == len(flags.file_list)

    coords_all, confs_all = load_predictions(flags.file_list, flags.weight)

    num_modes_total = confs_all.shape[1]
    n_example = len(coords_all)

    assert coords_all.shape == (n_example, num_modes_total, 50, 2)

    np_random = np.random.RandomState(SEED)

    ens_confs = np.zeros((n_example, 3))
    ens_coords = np.zeros((n_example, 3, 50, 2))
    ens_log_probs = np.zeros((n_example,))
    time_store = {}

    noise = np_random.normal(0, scale=flags.sigma, size=(flags.batch_size, flags.N_sample, 100))

    for idx in tqdm(range(0, n_example, flags.batch_size)):
        confidences = confs_all[idx:idx + flags.batch_size]
        confidences = confidences / confidences.sum(axis=1, keepdims=True)
        size = confidences.shape[0]
        assert confidences.shape == (size, num_modes_total)

        coords = coords_all[idx:idx + flags.batch_size]
        coords = coords.reshape(size * num_modes_total, 50 * 2)

        # TODO: remove for-loop
        with add_time("choice", time_store):
            indices = np.stack([np_random.choice(num_modes_total, size=flags.N_sample, p=confidences[j]) for j in range(size)], axis=0)
        indices = (np.arange(size)[:, np.newaxis] * num_modes_total + indices).reshape(-1)

        X = coords[indices]
        X = X.reshape(size, flags.N_sample, 100)
        X = X + noise[:len(X)]

        if flags.give_centroids_init:
            flags.gmm_kwargs["centroids_init"] = coords.reshape(size, num_modes_total, 100)[:, :3, :]

        if flags.precisions_init_sigma is not None:
            flags.gmm_kwargs["precisions_init"] = np.full((size, 3), flags.precisions_init_sigma)

        with add_time("fit", time_store):
            gmm = BatchSphericalGMM(n_components=3, device=flags.device, seed=SEED, **flags.gmm_kwargs)
            weights, means, _, log_probs = gmm.fit(X)

        ens_confs[idx:idx + flags.batch_size] = weights
        ens_coords[idx:idx + flags.batch_size] = means.reshape(means.shape[0], 3, 50, 2)
        ens_log_probs[idx:idx + flags.batch_size] = log_probs

    print(time_store)

    output_path = Path(flags.output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    if output_path.name.endswith(".csv"):
        save_format = "csv"
    else:
        save_format = "npz"

    timestamps, track_ids, targets, target_availabilities = load_metadata(flags.file_list)

    if targets is not None:
        assert target_availabilities is not None
        errors = pytorch_neg_multi_log_likelihood_batch(
            torch.as_tensor(targets),
            torch.as_tensor(ens_coords),
            torch.as_tensor(ens_confs),
            torch.as_tensor(target_availabilities),
        )
        print("errors", errors.shape, torch.mean(errors))

    if save_format == "csv":
        write_pred_csv(str(output_path), timestamps, track_ids, ens_coords, ens_confs)
    else:
        np.savez_compressed(
            str(output_path),
            timestamps=timestamps,
            track_ids=track_ids,
            coords=ens_coords,
            confs=ens_confs,
            targets=targets,
            target_availabilities=target_availabilities,
            log_probs=ens_log_probs,
        )
        print(f"Saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--yaml_filepath', '-y', type=str,
                        help='Flags yaml file path')
    args = parser.parse_args()

    with open(args.yaml_filepath, 'r') as f:
        flags = EnsembleFlags(**yaml.safe_load(f))

    ensemble_batch_core(flags)
