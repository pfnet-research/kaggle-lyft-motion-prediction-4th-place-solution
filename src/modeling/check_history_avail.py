"""
Calculate num history for chopped valid/test data...
"""
import argparse
from distutils.util import strtobool
import numpy as np
import torch
from pathlib import Path

from l5kit.dataset import AgentDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from l5kit.data import LocalDataManager, ChunkedDataset

import sys
import os

from tqdm import tqdm
sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))
from lib.dataset.faster_agent_dataset import FasterAgentDataset
from lib.evaluation.mask import load_mask_chopped
from modeling.load_flag import Flags
from lib.rasterization.rasterizer_builder import build_custom_rasterizer
from lib.utils.yaml_utils import save_yaml, load_yaml


def parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--out', '-o', default='results/tmp',
                        help='Directory to output the result')
    parser.add_argument('--debug', '-d', type=strtobool, default='false',
                        help='Debug mode')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()

    processed_dir = Path("../../input/processed_data")
    npz_path = processed_dir / f"history_avail.npz"
    print(f"Load from {npz_path}")
    results = np.load(npz_path)
    his_avail_valid = results["his_avail_valid"]
    his_avail_test = results["his_avail_test"]
    import IPython; IPython.embed()
    n_his_valid = np.sum(his_avail_valid, axis=1)
