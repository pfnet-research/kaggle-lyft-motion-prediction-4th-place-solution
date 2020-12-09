import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from numpy.random import multinomial, multivariate_normal
from sklearn.mixture import GaussianMixture
# from sklearn.mixture._gaussian_mixture import _estimate_gaussian_parameters
import warnings
import torch
warnings.simplefilter('ignore')

sys.path.append("./src")
sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))
# from lib.utils.timer_utils import timer_ms
from lib.mixture.gmm import GaussianMixtureIdentity
from lib.functions.nll import pytorch_neg_multi_log_likelihood_batch

NUM_VAL = 190327
# NUM_VAL = 5000
SEED = 0
np.random.seed(SEED)


def main(flags):
    outdir = flags["outdir"]
    if not Path(outdir).exists():
        # Calculate ensemble...
        w = flags["weight"]
        sigma = flags["sigma"]
        N_sample = flags["N_sample"]
        covariance_type = flags["covariance_type"]
        file_list = flags["file_list"]

        n_models: int = len(file_list)
        preds_list = [np.load(filepath) for filepath in file_list]
        # coords_array: (n_models, n_valid_data, num_modes=3, future_len=50, coords=2)
        coords_array = np.array([preds["coords"] for preds in preds_list])
        # confs_array: (n_models, n_valid_data, num_modes=3)
        confs_array = np.array([preds["confs"] for preds in preds_list])
        weighted_confs_array = np.array(w)[:, None, None] * confs_array
        weighted_confs_array = weighted_confs_array / np.sum(weighted_confs_array, axis=(0, 2), keepdims=True)
        print("coords_array", coords_array.shape, "confs_array", confs_array.shape)

        # 9164 ms
        # I = np.eye(100) * sigma
        # samples = [multivariate_normal(np.zeros(100), I) for _ in range(N_sample)]
        # print("samples", len(samples), "samples0", samples[0].shape, samples[0])
        # samples_list = samples

        # 12 ms
        if sigma > 0:
            samples = np.random.normal(0, np.sqrt(sigma), (N_sample, 100))
        else:
            samples = np.zeros((N_sample, 100), dtype=np.float64)
        print("samples", samples.shape, samples)

        coords_out = []
        confs_out = []

        for idx in tqdm(range(NUM_VAL)):
            # mu = [coords_array[i][idx].reshape(3, 100) for i in range(len(coords_array))]
            # mu = np.concatenate(mu)
            mu = coords_array[:, idx].reshape(n_models * 3, 100)
            # assert np.allclose(mu, mu2)
            # confidence = [w[i]*confs_array[i][idx] for i in range(len(confs_array))]
            # confidence = np.concatenate(confidence)
            # confidence /= confidence.sum()
            confidence = weighted_confs_array[:, idx].ravel()
            # assert np.allclose(confidence, confidence2)
            x = mu[np.random.choice(3*len(w), size=N_sample, p=confidence)]+samples
            if covariance_type == "identity":
                gauss = GaussianMixtureIdentity(3, "spherical", random_state=SEED)
            else:
                gauss = GaussianMixture(3, covariance_type, random_state=SEED)
                # if idx == 0:
                #     print("random")
                # gauss = GaussianMixture(3, covariance_type, random_state=SEED, init_params="random")
            gauss.fit(x)
            confidence_fit = gauss.weights_
            mu_fit = gauss.means_
            coords_out.append(mu_fit.reshape(3, 50, 2))
            confs_out.append(confidence_fit)

        preds0 = preds_list[0]
        timestamps = preds0["timestamps"]
        track_ids = preds0["track_ids"]
        targets = preds0["targets"]
        target_availabilities = preds0["target_availabilities"]
        coords = np.array(coords_out)
        confs = np.array(confs_out)
        np.savez_compressed(
            outdir,
            timestamps=timestamps,
            track_ids=track_ids,
            coords=coords,
            confs=confs,
            targets=targets,
            target_availabilities=target_availabilities,
        )
        print(f"Saved to {outdir}")
    else:
        # Just use already calculated results...
        preds = np.load(outdir)
        timestamps = preds["timestamps"]
        track_ids = preds["track_ids"]
        targets = preds["targets"]
        target_availabilities = preds["target_availabilities"]
        coords = preds["coords"]
        confs = preds["confs"]
    errors = pytorch_neg_multi_log_likelihood_batch(
        torch.as_tensor(targets)[:NUM_VAL],
        torch.as_tensor(coords),
        torch.as_tensor(confs),
        torch.as_tensor(target_availabilities)[:NUM_VAL],
        )
    print("errors", errors.shape, torch.mean(errors))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--yaml_filepath', '-y', type=str,
                        help='Flags yaml file path')
    args = parser.parse_args()
    with open(args.yaml_filepath, 'r') as f:
        flags = yaml.safe_load(f)
    main(flags)
