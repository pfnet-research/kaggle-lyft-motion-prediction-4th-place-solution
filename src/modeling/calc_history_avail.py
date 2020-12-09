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


def calc_history_avail(data_loader) -> np.ndarray:
    his_avail_list = []

    with torch.no_grad():
        dataiter = tqdm(data_loader)
        for data in dataiter:
            his_avail = data["history_availabilities"].numpy()
            # To reduce memory usage, convert to bool dtype.
            his_avail_list.append(his_avail.astype(np.bool))
    his_avail_array = np.concatenate(his_avail_list)
    return his_avail_array


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
    out_dir = Path(args.out)
    debug = args.debug

    flags_dict = load_yaml(out_dir / 'flags.yaml')
    cfg = load_yaml(out_dir / 'cfg.yaml')
    # flags = DotDict(flags_dict)
    flags = Flags()
    flags.update(flags_dict)
    print(f"flags: {flags_dict}")

    # set env variable for data
    # Not use flags.l5kit_data_folder, but use fixed test data.
    l5kit_data_folder = "../../input/lyft-motion-prediction-autonomous-vehicles"
    os.environ["L5KIT_DATA_FOLDER"] = l5kit_data_folder
    dm = LocalDataManager(None)

    print("Load dataset...")
    default_test_cfg = {
        'key': 'scenes/test.zarr',
        'batch_size': 32,
        'shuffle': False,
        'num_workers': 4
    }
    test_cfg = cfg.get("test_data_loader", default_test_cfg)

    # from copy import deepcopy
    # cfg2 = deepcopy(cfg)
    # cfg2["model_params"]["history_num_frames"] = 50
    cfg["model_params"]["history_num_frames"] = 100
    cfg["raster_params"]["map_type"] = "stub_debug"  # For faster calculation...

    # Rasterizer
    rasterizer = build_custom_rasterizer(cfg, dm)

    valid_cfg = cfg["valid_data_loader"]
    # valid_path = "scenes/sample.zarr" if debug else valid_cfg["key"]
    valid_path = valid_cfg["key"]
    valid_agents_mask = None
    if flags.validation_chopped:
        num_frames_to_chop = 100
        th_agent_prob = cfg["raster_params"]["filter_agents_threshold"]
        min_frame_future = 1
        num_frames_to_copy = num_frames_to_chop
        valid_agents_mask = load_mask_chopped(
            dm.require(valid_path), th_agent_prob, num_frames_to_copy, min_frame_future)
        print("valid_path", valid_path, "valid_agents_mask", valid_agents_mask.shape)
    valid_zarr = ChunkedDataset(dm.require(valid_path)).open(cached=False)
    valid_agent_dataset = FasterAgentDataset(
        cfg, valid_zarr, rasterizer, agents_mask=valid_agents_mask,
        min_frame_history=flags.min_frame_history, min_frame_future=flags.min_frame_future,
        override_sample_function_name=flags.override_sample_function_name,
    )
    # valid_dataset = TransformDataset(valid_agent_dataset, transform)
    valid_dataset = valid_agent_dataset

    # Only use `n_valid_data` dataset for fast check.
    # Sample dataset from regular interval, to increase variety/coverage
    n_valid_data = 150 if debug else -1
    print("n_valid_data", n_valid_data)
    if n_valid_data > 0 and n_valid_data < len(valid_dataset):
        valid_sub_indices = np.linspace(0, len(valid_dataset)-1, num=n_valid_data, dtype=np.int64)
        valid_dataset = Subset(valid_dataset, valid_sub_indices)
    valid_batchsize = valid_cfg["batch_size"]

    local_valid_dataset = valid_dataset
    collate_fn = None
    valid_loader = DataLoader(
        local_valid_dataset, valid_batchsize, shuffle=False,
        pin_memory=True, num_workers=valid_cfg["num_workers"],
        collate_fn=collate_fn)

    print(valid_agent_dataset)
    print("# AgentDataset test:", len(valid_agent_dataset))
    print("# ActualDataset test:", len(valid_dataset))
    in_channels, height, width = valid_agent_dataset[0]["image"].shape  # get input image shape
    print("in_channels", in_channels, "height", height, "width", width)

    # --- Calc valid data ---
    his_avail_valid = calc_history_avail(valid_loader)
    print("n_his_avail_valid", his_avail_valid.shape, his_avail_valid)

    # --- Calc test data ---
    test_path = test_cfg["key"]
    print(f"Loading from {test_path}")
    test_zarr = ChunkedDataset(dm.require(test_path)).open(cached=False)
    print("test_zarr", type(test_zarr))
    test_mask = np.load(f"{l5kit_data_folder}/scenes/mask.npz")["arr_0"]
    test_agent_dataset = AgentDataset(cfg, test_zarr, rasterizer, agents_mask=test_mask)
    test_dataset = test_agent_dataset
    if debug:
        # Only use 100 dataset for fast check...
        test_dataset = Subset(test_dataset, np.arange(100))
    test_loader = DataLoader(
        test_dataset,
        shuffle=test_cfg["shuffle"],
        batch_size=test_cfg["batch_size"],
        num_workers=test_cfg["num_workers"],
        pin_memory=True,
    )
    his_avail_test = calc_history_avail(test_loader)
    print("n_his_avail_test", his_avail_test.shape, his_avail_test)

    # --- Save to npz format, for future analysis purpose ---
    debug_str = "_debug" if debug else ""
    processed_dir = Path("../../input/processed_data")
    os.makedirs(str(processed_dir), exist_ok=True)

    npz_path = processed_dir / f"history_avail{debug_str}.npz"
    np.savez_compressed(
        npz_path,
        his_avail_valid=his_avail_valid,
        his_avail_test=his_avail_test,
    )
    print(f"Saved to {npz_path}")
    import IPython; IPython.embed()
