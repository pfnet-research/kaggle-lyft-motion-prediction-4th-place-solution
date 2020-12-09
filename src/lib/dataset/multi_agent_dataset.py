from functools import partial
from typing import Optional, cast, Tuple
import numpy as np

from l5kit.data import ChunkedDataset, get_frames_slice_from_scenes
from l5kit.dataset import EgoDataset
from l5kit.kinematic import Perturbation
from l5kit.rasterization import Rasterizer, RenderContext

import sys
import os
sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))
from lib.sampling.multi_agent_sampling import generate_multi_agent_sample


class MultiAgentDataset(EgoDataset):
    def __init__(
        self,
        cfg: dict,
        zarr_dataset: ChunkedDataset,
        rasterizer: Rasterizer,
        perturbation: Optional[Perturbation] = None,
        min_frame_history: int = 1,
        min_frame_future: int = 10,
    ):
        super(MultiAgentDataset, self).__init__(cfg, zarr_dataset, rasterizer, perturbation)

        render_context = RenderContext(
            raster_size_px=np.array(cfg["raster_params"]["raster_size"]),
            pixel_size_m=np.array(cfg["raster_params"]["pixel_size"]),
            center_in_raster_ratio=np.array(cfg["raster_params"]["ego_center"]),
        )

        self.sample_function = partial(
            generate_multi_agent_sample,
            render_context=render_context,
            history_num_frames=cfg["model_params"]["history_num_frames"],
            history_step_size=cfg["model_params"]["history_step_size"],
            future_num_frames=cfg["model_params"]["future_num_frames"],
            future_step_size=cfg["model_params"]["future_step_size"],
            filter_agents_threshold=cfg["raster_params"]["filter_agents_threshold"],
            rasterizer=rasterizer,
            perturbation=perturbation,
            min_frame_history=min_frame_history,
            min_frame_future=min_frame_future,
        )

    def get_frame(self, scene_index: int, state_index: int, track_id: Optional[int] = None) -> dict:
        """
        A utility function to get the rasterisation and trajectory target for a given agent in a given frame

        Args:
            scene_index (int): the index of the scene in the zarr
            state_index (int): a relative frame index in the scene
            track_id (Optional[int]): the agent to rasterize or None for the AV
        Returns:
            dict: the rasterised image, the target trajectory (position and yaw) along with their availability,
            the 2D matrix to center that agent, the agent track (-1 if ego) and the timestamp

        """
        frames = self.dataset.frames[get_frames_slice_from_scenes(self.dataset.scenes[scene_index])]
        data = self.sample_function(state_index, frames, self.dataset.agents, self.dataset.tl_faces, track_id)
        # 0,1,C -> C,0,1
        image = data["image"].transpose(2, 0, 1)

        target_positions = np.array(data["target_positions"], dtype=np.float32)
        target_yaws = np.array(data["target_yaws"], dtype=np.float32)

        history_positions = np.array(data["history_positions"], dtype=np.float32)
        history_yaws = np.array(data["history_yaws"], dtype=np.float32)

        timestamp = frames[state_index]["timestamp"]
        track_id = np.int64(-1 if track_id is None else track_id)  # always a number to avoid crashing torch

        return {
            "image": image,
            "target_positions": target_positions,
            "target_yaws": target_yaws,
            "target_availabilities": data["target_availabilities"],
            "history_positions": history_positions,
            "history_yaws": history_yaws,
            "history_availabilities": data["history_availabilities"],
            # "world_to_image": data["raster_from_world"],
            "raster_from_world": data["raster_from_world"],  # (3, 3)
            "track_id": track_id,
            "timestamp": timestamp,
            "centroid": data["centroid"],
            "yaw": data["yaw"],
            "extent": data["extent"],
            "track_ids": data["track_ids"],
            "centroid_pixel": data["centroid_pixel"].astype(np.int64),
        }
