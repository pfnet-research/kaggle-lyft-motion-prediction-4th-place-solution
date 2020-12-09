from typing import List, Optional, Tuple

import numpy as np

from l5kit.data import (
    TL_FACE_DTYPE,
    filter_agents_by_labels,
    filter_tl_faces_by_frames,
    get_agents_slice_from_frames,
    get_tl_faces_slice_from_frames,
)
from l5kit.data.filter import filter_agents_by_frames, filter_agents_by_track_id
from l5kit.dataset.agent import MIN_FRAME_HISTORY, MIN_FRAME_FUTURE
from l5kit.geometry import rotation33_as_yaw, transform_points, compute_agent_pose, transform_point, angular_distance
from l5kit.kinematic import Perturbation
from l5kit.rasterization import EGO_EXTENT_HEIGHT, EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, Rasterizer, RenderContext
from l5kit.sampling.slicing import get_future_slice, get_history_slice


def generate_multi_agent_sample(
    state_index: int,
    frames: np.ndarray,
    agents: np.ndarray,
    tl_faces: np.ndarray,
    selected_track_id: Optional[int],
    render_context: RenderContext,
    history_num_frames: int,
    history_step_size: int,
    future_num_frames: int,
    future_step_size: int,
    filter_agents_threshold: float,
    rasterizer: Optional[Rasterizer] = None,
    perturbation: Optional[Perturbation] = None,
    min_frame_history: int = MIN_FRAME_HISTORY,
    min_frame_future: int = MIN_FRAME_FUTURE,
) -> dict:
    """Generates the inputs and targets to train a deep prediction model. A deep prediction model takes as input
    the state of the world (here: an image we will call the "raster"), and outputs where that agent will be some
    seconds into the future.

    This function has a lot of arguments and is intended for internal use, you should try to use higher level classes
    and partials that use this function.

    Args:
        state_index (int): The anchor frame index, i.e. the "current" timestep in the scene
        frames (np.ndarray): The scene frames array, can be numpy array or a zarr array
        agents (np.ndarray): The full agents array, can be numpy array or a zarr array
        tl_faces (np.ndarray): The full traffic light faces array, can be numpy array or a zarr array
        selected_track_id (Optional[int]): Either None for AV, or the ID of an agent that you want to
        predict the future of. This agent is centered in the raster and the returned targets are derived from
        their future states.
        render_context (RenderContext):
            raster_size (Tuple[int, int]): Desired output raster dimensions
            pixel_size (np.ndarray): Size of one pixel in the real world
            ego_center (np.ndarray): Where in the raster to draw the ego, [0.5,0.5] would be the center
        history_num_frames (int): Amount of history frames to draw into the rasters
        history_step_size (int): Steps to take between frames, can be used to subsample history frames
        future_num_frames (int): Amount of history frames to draw into the rasters
        future_step_size (int): Steps to take between targets into the future
        filter_agents_threshold (float): Value between 0 and 1 to use as cutoff value for agent filtering
        based on their probability of being a relevant agent
        rasterizer (Optional[Rasterizer]): Rasterizer of some sort that draws a map image
        perturbation (Optional[Perturbation]): Object that perturbs the input and targets, used
to train models that can recover from slight divergence from training set data

    Raises:
        ValueError: A ValueError is returned if the specified ``selected_track_id`` is not present in the scene
        or was filtered by applying the ``filter_agent_threshold`` probability filtering.

    Returns:
        dict: a dict object with the raster array, the future offset coordinates (meters),
        the future yaw angular offset, the future_availability as a binary mask
    """
    #  the history slice is ordered starting from the latest frame and goes backward in time., ex. slice(100, 91, -2)
    history_slice = get_history_slice(state_index, history_num_frames, history_step_size, include_current_state=True)
    future_slice = get_future_slice(state_index, future_num_frames, future_step_size)

    history_frames = frames[history_slice].copy()  # copy() required if the object is a np.ndarray
    future_frames = frames[future_slice].copy()

    sorted_frames = np.concatenate((history_frames[::-1], future_frames))  # from past to future

    # get agents (past and future)
    agent_slice = get_agents_slice_from_frames(sorted_frames[0], sorted_frames[-1])
    agents = agents[agent_slice].copy()  # this is the minimum slice of agents we need
    history_frames["agent_index_interval"] -= agent_slice.start  # sync interval with the agents array
    future_frames["agent_index_interval"] -= agent_slice.start  # sync interval with the agents array
    history_agents = filter_agents_by_frames(history_frames, agents)
    future_agents = filter_agents_by_frames(future_frames, agents)

    try:
        tl_slice = get_tl_faces_slice_from_frames(history_frames[-1], history_frames[0])  # -1 is the farthest
        # sync interval with the traffic light faces array
        history_frames["traffic_light_faces_index_interval"] -= tl_slice.start
        history_tl_faces = filter_tl_faces_by_frames(history_frames, tl_faces[tl_slice].copy())
    except ValueError:
        history_tl_faces = [np.empty(0, dtype=TL_FACE_DTYPE) for _ in history_frames]

    if perturbation is not None:
        history_frames, future_frames = perturbation.perturb(
            history_frames=history_frames, future_frames=future_frames
        )

    # State you want to predict the future of.
    cur_frame = history_frames[0]
    cur_agents = history_agents[0]

    cur_agents = filter_agents_by_labels(cur_agents, filter_agents_threshold)
    agent_track_ids_u64 = cur_agents["track_id"]
    # uint64 --> int64
    agent_track_ids = agent_track_ids_u64.astype(np.int64)
    assert np.alltrue(agent_track_ids == agent_track_ids_u64)
    agent_track_ids = np.concatenate([np.array([-1], dtype=np.int64), agent_track_ids])

    # Draw image with Ego car in center
    selected_agent = None
    input_im = (
        None
        if not rasterizer
        else rasterizer.rasterize(history_frames, history_agents, history_tl_faces, selected_agent)
    )

    future_coords_offset_list = []
    future_yaws_offset_list = []
    future_availability_list = []
    history_coords_offset_list = []
    history_yaws_offset_list = []
    history_availability_list = []
    agent_centroid_list = []
    agent_yaw_list = []
    agent_extent_list = []
    filtered_track_ids_list = []
    for selected_track_id in agent_track_ids:
        if selected_track_id == -1:
            agent_centroid = cur_frame["ego_translation"][:2]
            agent_yaw_rad = rotation33_as_yaw(cur_frame["ego_rotation"])
            agent_extent = np.asarray((EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, EGO_EXTENT_HEIGHT))

            world_from_agent = compute_agent_pose(agent_centroid, agent_yaw_rad)
            agent_from_world = np.linalg.inv(world_from_agent)
            raster_from_world = render_context.raster_from_world(agent_centroid, agent_yaw_rad)

            agent_origin = np.zeros((2,), dtype=np.float32)
        else:
            # this will raise IndexError if the agent is not in the frame or under agent-threshold
            # this is a strict error, we cannot recover from this situation
            try:
                agent = filter_agents_by_track_id(cur_agents, selected_track_id)[0]
            except IndexError:
                raise ValueError(f" track_id {selected_track_id} not in frame or below threshold")
            agent_centroid = agent["centroid"]
            agent_yaw_rad = agent["yaw"]
            agent_extent = agent["extent"]

            agent_origin = transform_point(agent_centroid, agent_from_world)
        future_coords_offset, future_yaws_offset, future_availability = _create_targets_for_deep_prediction(
            future_num_frames, future_frames, selected_track_id, future_agents, agent_from_world, agent_yaw_rad,
            agent_origin
        )
        if selected_track_id != -1 and np.sum(future_availability) < min_frame_future:
            # Not enough future to predict, skip this agent.
            continue
        # history_num_frames + 1 because it also includes the current frame
        history_coords_offset, history_yaws_offset, history_availability = _create_targets_for_deep_prediction(
            history_num_frames + 1, history_frames, selected_track_id, history_agents, agent_from_world, agent_yaw_rad,
            agent_origin
        )
        if selected_track_id != -1 and np.sum(history_availability) < min_frame_history:
            # Not enough history to predict, skip this agent.
            continue
        future_coords_offset_list.append(future_coords_offset)
        future_yaws_offset_list.append(future_yaws_offset)
        future_availability_list.append(future_availability)
        history_coords_offset_list.append(history_coords_offset)
        history_yaws_offset_list.append(history_yaws_offset)
        history_availability_list.append(history_availability)
        agent_centroid_list.append(agent_centroid)
        agent_yaw_list.append(agent_yaw_rad)
        agent_extent_list.append(agent_extent)
        filtered_track_ids_list.append(selected_track_id)

    # Get pixel coordinate
    agent_centroid_array = np.array(agent_centroid_list)
    agent_centroid_in_pixel = transform_points(agent_centroid_array, raster_from_world)

    return {
        "image": input_im,  # (h, w, ch)
        # --- All below is in world coordinate ---
        "target_positions": np.array(future_coords_offset_list),  # (n_agents, num_frames, 2)
        "target_yaws": np.array(future_yaws_offset_list),  # (n_agents, num_frames, 1)
        "target_availabilities": np.array(future_availability_list),  # (n_agents, num_frames)
        "history_positions": np.array(history_coords_offset_list),  # (n_agents, num_frames, 2)
        "history_yaws": np.array(history_yaws_offset_list),  # (n_agents, num_frames, 1)
        "history_availabilities": np.array(history_availability_list),  # (n_agents, num_frames)
        # "world_to_image": raster_from_world,  # (3, 3)
        "raster_from_world": raster_from_world,  # (3, 3)
        "centroid": agent_centroid_array,  # (n_agents, 2)
        "yaw": np.array(agent_yaw_list),  # (n_agents, 1)
        "extent": np.array(agent_extent_list),  # (n_agents, 3)
        "track_ids": np.array(filtered_track_ids_list),  # (n_agents)
        "centroid_pixel": agent_centroid_in_pixel,  # (n_agents, 2)
    }


def _create_targets_for_deep_prediction(
    num_frames: int,
    frames: np.ndarray,
    selected_track_id: int,  # modified that ego car is -1, not None (None is used in l5kit)
    agents: List[np.ndarray],
    agent_from_world: np.ndarray,
    current_agent_yaw: float,
    agent_origin: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Internal function that creates the targets and availability masks for deep prediction-type models.
    The futures/history offset (in meters) are computed. When no info is available (e.g. agent not in frame)
    a 0 is set in the availability array (1 otherwise).

    Args:
        num_frames (int): number of offset we want in the future/history
        frames (np.ndarray): available frames. This may be less than num_frames
        selected_track_id (Optional[int]): agent track_id or AV (-1)
        agents (List[np.ndarray]): list of agents arrays (same len of frames)
        agent_from_world (np.ndarray): local from world matrix
        current_agent_yaw (float): angle of the agent at timestep 0
        agent_origin (np.ndarray): (2,) xy coord

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: position offsets, angle offsets, availabilities

    """
    # How much the coordinates differ from the current state in meters.
    coords_offset = np.zeros((num_frames, 2), dtype=np.float32)
    yaws_offset = np.zeros((num_frames, 1), dtype=np.float32)
    availability = np.zeros((num_frames,), dtype=np.float32)

    for i, (frame, agents) in enumerate(zip(frames, agents)):
        if selected_track_id == -1:
            agent_centroid = frame["ego_translation"][:2]
            agent_yaw = rotation33_as_yaw(frame["ego_rotation"])
        else:
            # it's not guaranteed the target will be in every frame
            try:
                agent = filter_agents_by_track_id(agents, selected_track_id)[0]
            except IndexError:
                availability[i] = 0.0  # keep track of invalid futures/history
                continue

            agent_centroid = agent["centroid"]
            agent_yaw = agent["yaw"]

        coords_offset[i] = transform_point(agent_centroid, agent_from_world) - agent_origin
        yaws_offset[i] = angular_distance(agent_yaw, current_agent_yaw)
        availability[i] = 1.0
    return coords_offset, yaws_offset, availability
