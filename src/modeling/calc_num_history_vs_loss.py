"""
Calculate num history for chopped valid/test data...
"""
import argparse
from distutils.util import strtobool
import numpy as np
import torch
from pathlib import Path

# from l5kit.dataset import AgentDataset
# from torch.utils.data import DataLoader
# from torch.utils.data.dataset import Subset
#
# from l5kit.data import LocalDataManager, ChunkedDataset

import sys
import os

from tqdm import tqdm
sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))
from lib.functions.nll import pytorch_neg_multi_log_likelihood_batch
# from lib.dataset.faster_agent_dataset import FasterAgentDataset
# from lib.evaluation.mask import load_mask_chopped
# from modeling.load_flag import Flags
# from lib.rasterization.rasterizer_builder import build_custom_rasterizer
# from lib.utils.yaml_utils import save_yaml, load_yaml


def parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--pred_npz_path', '-p', default='results/tmp/eval_ema/pred.npz',
                        help='pred.npz filepath')
    parser.add_argument('--debug', '-d', type=strtobool, default='false',
                        help='Debug mode')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    debug = args.debug
    device = args.device

    # --- Load n_availability ---
    processed_dir = Path("../../input/processed_data")
    npz_path = processed_dir / f"n_history.npz"
    print(f"Load from {npz_path}")
    n_his = np.load(npz_path)
    n_his_avail_valid = n_his["n_his_avail_valid"]
    n_his_avail_test = n_his["n_his_avail_test"]

    # --- Load pred ---
    preds = np.load(args.pred_npz_path)
    coords = preds["coords"]
    confs = preds["confs"]
    targets = preds["targets"]
    target_availabilities = preds["target_availabilities"]

    # Evaluate loss
    errors = pytorch_neg_multi_log_likelihood_batch(
        torch.as_tensor(targets, device=device),
        torch.as_tensor(coords, device=device),
        torch.as_tensor(confs, device=device),
        torch.as_tensor(target_availabilities, device=device),
        reduction="none")
    print("errors", errors.shape, torch.mean(errors))

    n_his_avail_valid = torch.as_tensor(n_his_avail_valid.astype(np.int64), device=device)
    for i in range(1, 11):
        this_error = errors[n_his_avail_valid == i]
        mean_error = torch.mean(this_error)
        print(f"i=={i:4.0f}: {mean_error:10.4f}, {len(this_error)}")

    for i in [20, 50, 100]:
        this_error = errors[n_his_avail_valid >= i]
        mean_error = torch.mean(this_error)
        print(f"i>={i:4.0f}: {mean_error:10.4f}, {len(this_error)}")
    import IPython; IPython.embed()
