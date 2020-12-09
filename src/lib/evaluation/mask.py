import argparse
import os
from pathlib import Path

import numpy as np
from zarr import convenience

from l5kit.data import ChunkedDataset, get_agents_slice_from_frames
from l5kit.dataset.select_agents import TH_DISTANCE_AV, TH_EXTENT_RATIO, TH_YAW_DEGREE, select_agents

MIN_FUTURE_STEPS = 10


def get_mask_chopped_path(
    zarr_path: str, th_agent_prob: float, num_frames_to_copy: int, min_frame_future: int
) -> Path:
    zarr_path = Path(zarr_path)
    dest_path = zarr_path.parent / f"{zarr_path.stem}_chopped_valid"
    os.makedirs(str(dest_path), exist_ok=True)
    mask_chopped_path = dest_path / f"mask_{th_agent_prob}_{num_frames_to_copy}_{min_frame_future}.npz"
    return mask_chopped_path


def create_chopped_mask(
    zarr_path: str, th_agent_prob: float, num_frames_to_copy: int, min_frame_future: int
) -> str:
    """Create mask to emulate chopped dataset with gt data.

    Args:
        zarr_path (str): input zarr path to be chopped
        th_agent_prob (float): threshold over agents probabilities used in select_agents function
        num_frames_to_copy (int):  number of frames to copy from the beginning of each scene, others will be discarded
        min_frame_future (int): minimum number of frames that must be available in the future for an agent

    Returns:
        str: Path to saved mask
    """
    zarr_path = Path(zarr_path)
    mask_chopped_path = get_mask_chopped_path(zarr_path, th_agent_prob, num_frames_to_copy, min_frame_future)

    # Create standard mask for the dataset so we can use it to filter out unreliable agents
    zarr_dt = ChunkedDataset(str(zarr_path))
    zarr_dt.open()

    agents_mask_path = Path(zarr_path) / f"agents_mask/{th_agent_prob}"
    if not agents_mask_path.exists():  # don't check in root but check for the path
        select_agents(
            zarr_dt,
            th_agent_prob=th_agent_prob,
            th_yaw_degree=TH_YAW_DEGREE,
            th_extent_ratio=TH_EXTENT_RATIO,
            th_distance_av=TH_DISTANCE_AV,
        )
    agents_mask_origin = np.asarray(convenience.load(str(agents_mask_path)))

    # compute the chopped boolean mask, but also the original one limited to frames of interest for GT csv
    agents_mask_orig_bool = np.zeros(len(zarr_dt.agents), dtype=np.bool)

    for idx in range(len(zarr_dt.scenes)):
        scene = zarr_dt.scenes[idx]

        frame_original = zarr_dt.frames[scene["frame_index_interval"][0] + num_frames_to_copy - 1]
        slice_agents_original = get_agents_slice_from_frames(frame_original)

        mask = agents_mask_origin[slice_agents_original][:, 1] >= min_frame_future
        agents_mask_orig_bool[slice_agents_original] = mask.copy()

    # store the mask and the GT csv of frames on interest
    np.savez(str(mask_chopped_path), agents_mask_orig_bool)
    return str(mask_chopped_path)


def load_mask_chopped(
    zarr_path: str, th_agent_prob: float, num_frames_to_copy: int, min_frame_future: int
) -> np.ndarray:
    mask_chopped_path = get_mask_chopped_path(
        zarr_path, th_agent_prob, num_frames_to_copy, min_frame_future)
    if not mask_chopped_path.exists():
        print(f"Cache not exist, creating {mask_chopped_path}")
        mask_chopped_path2 = create_chopped_mask(zarr_path, th_agent_prob, num_frames_to_copy, min_frame_future)
        assert str(mask_chopped_path) == str(mask_chopped_path2)
    agents_mask_orig_bool = np.load(str(mask_chopped_path))["arr_0"]
    return agents_mask_orig_bool
