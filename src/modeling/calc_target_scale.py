from typing import Tuple

import dataclasses
import numpy as np
import torch
from pathlib import Path

from l5kit.data import LocalDataManager, ChunkedDataset
import sys
import os

from tqdm import tqdm

sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))
from lib.evaluation.mask import load_mask_chopped
from lib.rasterization.rasterizer_builder import build_custom_rasterizer
from lib.dataset.faster_agent_dataset import FasterAgentDataset
from lib.utils.yaml_utils import save_yaml, load_yaml
from modeling.load_flag import load_flags, Flags


def calc_target_scale(agent_dataset, n_sample: int = 10000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub_indices = np.linspace(0, len(agent_dataset) - 1, num=n_sample, dtype=np.int64)
    pos_list = []
    for i in tqdm(sub_indices):
        d = agent_dataset[i]
        pos = d["target_positions"]
        pos[~d["target_availabilities"].astype(bool)] = np.nan
        pos_list.append(pos)
    agents_pos = np.array(pos_list)
    target_scale_abs_mean = np.nanmean(np.abs(agents_pos), axis=0)
    target_scale_abs_max = np.nanmax(np.abs(agents_pos), axis=0)
    target_scale_std = np.nanstd(agents_pos, axis=0)
    return target_scale_abs_mean, target_scale_abs_max, target_scale_std


if __name__ == '__main__':
    mode = ""
    flags: Flags = load_flags(mode=mode)
    flags_dict = dataclasses.asdict(flags)
    cfg = load_yaml(flags.cfg_filepath)
    out_dir = Path(flags.out_dir)
    print(f"cfg {cfg}")
    os.makedirs(str(out_dir), exist_ok=True)
    print(f"flags: {flags_dict}")
    save_yaml(out_dir / 'flags.yaml', flags_dict)
    save_yaml(out_dir / 'cfg.yaml', cfg)
    debug = flags.debug

    # set env variable for data
    os.environ["L5KIT_DATA_FOLDER"] = flags.l5kit_data_folder
    dm = LocalDataManager(None)

    print("init dataset")
    train_cfg = cfg["train_data_loader"]
    valid_cfg = cfg["valid_data_loader"]

    # Build StubRasterizer for fast dataset access
    cfg["raster_params"]["map_type"] = "stub_debug"
    rasterizer = build_custom_rasterizer(cfg, dm)
    print("rasterizer", rasterizer)

    train_path = "scenes/sample.zarr" if debug else train_cfg["key"]

    train_agents_mask = None
    if flags.validation_chopped:
        # Use chopped dataset to calc statistics...
        num_frames_to_chop = 100
        th_agent_prob = cfg["raster_params"]["filter_agents_threshold"]
        min_frame_future = 1
        num_frames_to_copy = num_frames_to_chop
        train_agents_mask = load_mask_chopped(
            dm.require(train_path), th_agent_prob, num_frames_to_copy, min_frame_future)
        print("train_path", train_path, "train_agents_mask", train_agents_mask.shape)

    train_zarr = ChunkedDataset(dm.require(train_path)).open(cached=False)
    print("train_zarr", type(train_zarr))
    print(f"Open Dataset {flags.pred_mode}...")

    train_agent_dataset = FasterAgentDataset(
        cfg, train_zarr, rasterizer, min_frame_history=flags.min_frame_history,
        min_frame_future=flags.min_frame_future, agents_mask=train_agents_mask
    )
    print("train_agent_dataset", len(train_agent_dataset))
    n_sample = 1_000_000  # Take 1M sample.
    target_scale_abs_mean, target_scale_abs_max, target_scale_std = calc_target_scale(train_agent_dataset, n_sample)

    chopped_str = "_chopped" if flags.validation_chopped else ""
    agent_prob = cfg["raster_params"]["filter_agents_threshold"]
    filename = f"target_scale_abs_mean_{agent_prob}_{flags.min_frame_history}_{flags.min_frame_future}{chopped_str}.npz"
    cache_path = Path(train_zarr.path) / filename
    np.savez_compressed(cache_path, target_scale=target_scale_abs_mean)
    print("Saving to ", cache_path)

    filename = f"target_scale_abs_max_{agent_prob}_{flags.min_frame_history}_{flags.min_frame_future}{chopped_str}.npz"
    cache_path = Path(train_zarr.path) / filename
    np.savez_compressed(cache_path, target_scale=target_scale_abs_max)
    print("Saving to ", cache_path)

    filename = f"target_scale_std_{agent_prob}_{flags.min_frame_history}_{flags.min_frame_future}{chopped_str}.npz"
    cache_path = Path(train_zarr.path) / filename
    np.savez_compressed(cache_path, target_scale=target_scale_std)
    print("Saving to ", cache_path)

    print("target_scale_abs_mean", target_scale_abs_mean)
    print("target_scale_abs_max", target_scale_abs_max)
    print("target_scale_std", target_scale_std)
    import IPython; IPython.embed()
