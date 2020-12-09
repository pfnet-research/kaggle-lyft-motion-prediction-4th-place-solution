"""
Check Rasterizer behavior for valid data...
"""
import argparse
from copy import deepcopy
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
from lib.utils.timer_utils import timer
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

    # Rasterizer
    rasterizer = build_custom_rasterizer(cfg, dm)

    tuned_cfg = deepcopy(cfg)
    assert tuned_cfg["raster_params"]["map_type"] == "py_semantic"
    # tuned_cfg["raster_params"]["map_type"] = "tuned_box+semantic_debug"
    tuned_cfg["raster_params"]["map_type"] = "tuned_box+tuned_semantic"
    tuned_rasterizer = build_custom_rasterizer(tuned_cfg, dm)
    print("tuned_rasterizer", tuned_rasterizer)

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
    tuned_valid_agent_dataset = FasterAgentDataset(
        tuned_cfg, valid_zarr, tuned_rasterizer, agents_mask=valid_agents_mask,
        min_frame_history=flags.min_frame_history, min_frame_future=flags.min_frame_future,
        override_sample_function_name=flags.override_sample_function_name,
    )
    # valid_dataset = TransformDataset(valid_agent_dataset, transform)
    valid_dataset = valid_agent_dataset
    tuned_valid_dataset = tuned_valid_agent_dataset

    # Only use `n_valid_data` dataset for fast check.
    # Sample dataset from regular interval, to increase variety/coverage
    n_valid_data = 150 if debug else -1
    print("n_valid_data", n_valid_data)
    if n_valid_data > 0 and n_valid_data < len(valid_dataset):
        valid_sub_indices = np.linspace(0, len(valid_dataset)-1, num=n_valid_data, dtype=np.int64)
        valid_dataset = Subset(valid_dataset, valid_sub_indices)
        tuned_valid_dataset = Subset(tuned_valid_dataset, valid_sub_indices)
    valid_batchsize = valid_cfg["batch_size"]

    collate_fn = None
    valid_loader = DataLoader(
        valid_dataset, valid_batchsize, shuffle=False,
        pin_memory=True, num_workers=valid_cfg["num_workers"],
        collate_fn=collate_fn)
    tuned_valid_loader = DataLoader(
        tuned_valid_dataset, valid_batchsize, shuffle=False,
        pin_memory=True, num_workers=valid_cfg["num_workers"],
        collate_fn=collate_fn)

    print(valid_agent_dataset)
    print("# AgentDataset test:", len(valid_agent_dataset))
    print("# ActualDataset test:", len(valid_dataset))
    in_channels, height, width = valid_agent_dataset[0]["image"].shape  # get input image shape
    print("in_channels", in_channels, "height", height, "width", width)

    # ----- check speed -----
    # --- Calc data by indexing ---
    n_data = 20 if debug else 200
    with timer("tuned data1"):
        tuned_data_list = [tuned_valid_dataset[i] for i in range(n_data)]
    with timer("tuned data2"):
        tuned_data_list = [tuned_valid_dataset[i] for i in range(n_data)]
    with timer("original data1"):
        data_list = [valid_dataset[i] for i in range(n_data)]
    with timer("original data2"):
        data_list = [valid_dataset[i] for i in range(n_data)]

    print("Checking data equal...")
    for i in tqdm(range(n_data)):
        if not np.allclose(data_list[i]["image"], tuned_data_list[i]["image"], rtol=1e-1, atol=1e-1):
            print(f"allclose failed!!! at i={i}")
            import IPython; IPython.embed()
            raise ValueError()

    # --- Calc data by loader ---
    batch_list = []
    tuned_batch_list = []

    n_iter = 2 if debug else 20
    with timer("tuned loader"):
        for data in tqdm(tuned_valid_loader):
            tuned_batch_list.append(data)
            if len(tuned_batch_list) >= n_iter:
                break

    with timer("original loader"):
        for data in tqdm(valid_loader):
            batch_list.append(data)
            if len(batch_list) >= n_iter:
                break

    # ----- Check data & tuned_data is same -----
    print("Checking batch equal...")
    for i in tqdm(range(n_iter)):
        if not torch.allclose(batch_list[i]["image"], tuned_batch_list[i]["image"], rtol=1e-1, atol=1e-1):
            print(f"allclose failed!!! at i={i}")
            import IPython; IPython.embed()

    import IPython; IPython.embed()

    # --- These block does not work well, since data loader prefetches data... ---
    # valid_iter = valid_loader.__iter__()
    # tuned_valid_iter = tuned_valid_loader.__iter__()
    # for i in range(10):
    #     print(f"i={i}")
    #     with timer("original"):
    #         batch = valid_iter.next()
    #     with timer("tuned"):
    #         tuned_batch = tuned_valid_iter.next()
    #
    #     # Check data & tuned_data is same...
    #     import IPython; IPython.embed()

    # # --- Calc test data ---
    # test_path = test_cfg["key"]
    # print(f"Loading from {test_path}")
    # test_zarr = ChunkedDataset(dm.require(test_path)).open(cached=False)
    # print("test_zarr", type(test_zarr))
    # test_mask = np.load(f"{l5kit_data_folder}/scenes/mask.npz")["arr_0"]
    # test_agent_dataset = AgentDataset(cfg, test_zarr, rasterizer, agents_mask=test_mask)
    # test_dataset = test_agent_dataset
    # if debug:
    #     # Only use 100 dataset for fast check...
    #     test_dataset = Subset(test_dataset, np.arange(100))
    # test_loader = DataLoader(
    #     test_dataset,
    #     shuffle=test_cfg["shuffle"],
    #     batch_size=test_cfg["batch_size"],
    #     num_workers=test_cfg["num_workers"],
    #     pin_memory=True,
    # )
    # n_his_avail_test = calc_num_history(test_loader)
    # print("n_his_avail_test", n_his_avail_test.shape, n_his_avail_test)
