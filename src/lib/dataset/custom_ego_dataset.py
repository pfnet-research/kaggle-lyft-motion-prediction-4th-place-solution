import warnings
from typing import Optional
import numpy as np

from l5kit.data import get_frames_slice_from_scenes


def get_frame_custom(self, scene_index: int, state_index: int, track_id: Optional[int] = None) -> dict:
    """Customized `get_frame` function, which returns all `data` entries of `sample_function`.
    A utility function to get the rasterisation and trajectory target for a given agent in a given frame

    Args:
        self: Ego Dataset
        scene_index (int): the index of the scene in the zarr
        state_index (int): a relative frame index in the scene
        track_id (Optional[int]): the agent to rasterize or None for the AV
    Returns:
        dict: the rasterised image, the target trajectory (position and yaw) along with their availability,
        the 2D matrix to center that agent, the agent track (-1 if ego) and the timestamp

    """
    frames = self.dataset.frames[get_frames_slice_from_scenes(self.dataset.scenes[scene_index])]

    tl_faces = self.dataset.tl_faces
    try:
        if self.cfg["raster_params"]["disable_traffic_light_faces"]:
            tl_faces = np.empty(0, dtype=self.dataset.tl_faces.dtype)  # completely disable traffic light faces
    except KeyError:
        warnings.warn(
            "disable_traffic_light_faces not found in config, this will raise an error in the future",
            RuntimeWarning,
            stacklevel=2,
        )
    data = self.sample_function(state_index, frames, self.dataset.agents, tl_faces, track_id)
    # 0,1,C -> C,0,1
    image = data["image"].transpose(2, 0, 1)

    target_positions = np.array(data["target_positions"], dtype=np.float32)
    target_yaws = np.array(data["target_yaws"], dtype=np.float32)

    history_positions = np.array(data["history_positions"], dtype=np.float32)
    history_yaws = np.array(data["history_yaws"], dtype=np.float32)

    timestamp = frames[state_index]["timestamp"]
    track_id = np.int64(-1 if track_id is None else track_id)  # always a number to avoid crashing torch

    data["image"] = image
    data["target_positions"] = target_positions
    data["target_yaws"] = target_yaws
    data["history_positions"] = history_positions
    data["history_yaws"] = history_yaws
    data["track_id"] = track_id
    data["timestamp"] = timestamp
    return data
    # return {
    #     "image": image,
    #     "target_positions": target_positions,
    #     "target_yaws": target_yaws,
    #     "target_availabilities": data["target_availabilities"],
    #     "history_positions": history_positions,
    #     "history_yaws": history_yaws,
    #     "history_availabilities": data["history_availabilities"],
    #     "world_to_image": data["raster_from_world"],  # TODO deprecate
    #     "raster_from_world": data["raster_from_world"],
    #     "raster_from_agent": data["raster_from_agent"],
    #     "agent_from_world": data["agent_from_world"],
    #     "world_from_agent": data["world_from_agent"],
    #     "track_id": track_id,
    #     "timestamp": timestamp,
    #     "centroid": data["centroid"],
    #     "yaw": data["yaw"],
    #     "extent": data["extent"],
    # }
