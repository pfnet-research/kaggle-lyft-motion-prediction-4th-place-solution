import copy
import os
from functools import partial
from typing import Optional, Tuple, Union, List

import numpy as np
from l5kit.data import (
    filter_agents_by_labels,
    filter_tl_faces_by_frames,
    get_agents_slice_from_frames,
    get_tl_faces_slice_from_frames,
)
from l5kit.data.filter import filter_agents_by_frames, filter_agents_by_track_id
from l5kit.data.labels import PERCEPTION_LABELS
from l5kit.data import LocalDataManager
from l5kit.geometry import compute_agent_pose, rotation33_as_yaw
from l5kit.rasterization import EGO_EXTENT_HEIGHT, EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, Rasterizer, RenderContext
from l5kit.rasterization.semantic_rasterizer import elements_within_bounds, SemanticRasterizer
from l5kit.rasterization.rasterizer_builder import build_rasterizer
from l5kit.sampling.slicing import get_future_slice, get_history_slice
from l5kit.sampling.agent_sampling import _create_targets_for_deep_prediction
from l5kit.kinematic import Perturbation
from shapely.geometry import Polygon
from shapely.geometry import Point


def create_generate_agent_sample_fixing_yaw_partial(cfg, rasterizer):
    render_context = RenderContext(
        raster_size_px=np.array(cfg["raster_params"]["raster_size"]),
        pixel_size_m=np.array(cfg["raster_params"]["pixel_size"]),
        center_in_raster_ratio=np.array(cfg["raster_params"]["ego_center"]),
    )

    # SemanticRasterizer の一部機能を作るため build する
    # TODO(kaizaburo): LocalDataManager を外から受け取るか、そもそも SemanticRasterizer　を使わない
    assert "L5KIT_DATA_FOLDER" in os.environ
    sat_rast_cfg = copy.deepcopy(cfg)
    sat_rast_cfg["raster_params"]["map_type"] = "semantic_debug"
    sat_rast: SemanticRasterizer = build_rasterizer(sat_rast_cfg, LocalDataManager(None))

    # Tuned by optuna
    default_yaw_fix_params = {
        "angle_diff_th_in_degree": 30.00639299068446,
        "distance_th_in_agent": 0.031291713326932744,
        "num_use_history": 4,
        "lane_polygon_buffer": 0.8153613050209113
    }

    yaw_fix_params = cfg["raster_params"].get("yaw_fix_params", default_yaw_fix_params)

    sample_function = partial(
        generate_agent_sample_fixing_yaw,
        render_context=render_context,
        history_num_frames=cfg["model_params"]["history_num_frames"],
        history_step_size=cfg["model_params"]["history_step_size"],
        future_num_frames=cfg["model_params"]["future_num_frames"],
        future_step_size=cfg["model_params"]["future_step_size"],
        filter_agents_threshold=cfg["raster_params"]["filter_agents_threshold"],
        sat_rast=sat_rast,
        rasterizer=rasterizer,
        perturbation=None,
        angle_diff_th_in_degree=yaw_fix_params["angle_diff_th_in_degree"],
        distance_th_in_agent=yaw_fix_params["distance_th_in_agent"],
        num_use_history=yaw_fix_params["num_use_history"],
        lane_polygon_buffer=yaw_fix_params["lane_polygon_buffer"],
    )
    return sample_function


def _angle_diff(a: np.ndarray, b: np.ndarray):
    diff = np.mod(a - b, 2 * np.pi)
    return np.minimum(diff, np.abs(diff - 2 * np.pi))


def _vec_to_rad(vec: Union[Tuple[float, float], np.ndarray]):
    dx, dy = vec
    if dx < 0:
        return np.pi + np.arctan(dy / dx)
    else:
        return np.arctan(dy / dx)


def fix_yaw_by_surrounding_lane_and_history(
    sat_rast: SemanticRasterizer,
    centroid: np.ndarray,
    yaw_origin: float,
    history_positions: np.ndarray,
    angle_diff_th_in_degree: float = 60,
    distance_th_in_agent: float = 0.5,
    num_use_history: int = 2,
    lane_polygon_buffer: float = 1.5,
) -> Optional[float]:
    """
    Args:
        sat_rast:
        centroid: (x, y) in world coord
        yaw_origin:
        history_positions:
    """
    p = Point(centroid)

    yaws = []
    on_road = False
    for k, idx in enumerate(elements_within_bounds(centroid, sat_rast.bounds_info["lanes"]["bounds"], 0)):
        lane_coords = sat_rast.proto_API.get_lane_coords(sat_rast.bounds_info["lanes"]["ids"][idx])
        coords = np.vstack((lane_coords["xyz_left"][:, :2], np.flip(lane_coords["xyz_right"][:, :2], 0)))
        lane_polygon = Polygon(coords)
        if lane_polygon.contains(p):
            on_road = True
        lane_polygon = lane_polygon.exterior.buffer(lane_polygon_buffer).union(lane_polygon)
        if lane_polygon.contains(p):
            min_coord_index = np.argmin(np.linalg.norm(lane_coords["xyz_left"][:, :2] - centroid, axis=1))
            if min_coord_index == lane_coords["xyz_left"].shape[0] - 1:
                min_coord_index -= 1
                assert min_coord_index >= 0
            yaw = _vec_to_rad(
                lane_coords["xyz_left"][min_coord_index + 1][:2]
                - lane_coords["xyz_left"][min_coord_index][:2]
            )
            yaws.append(yaw)
    yaws = np.array(yaws)

    # 駐車されている車を避けるため、道路に乗っていないものは弾く
    if not on_road:
        return None

    # まず history から yaw を作ることを考える (5とかは適当な定数)
    if len(history_positions) >= num_use_history:
        vecs = history_positions[:-1] - history_positions[1:]
        distances = np.linalg.norm(vecs, axis=1)
        if (distances[:num_use_history - 1] >= distance_th_in_agent).all():
            yaw = _vec_to_rad(vecs[0]) + yaw_origin
            # yaw_origin でうまく行っている
            if angle_diff_th_in_degree / 180 * np.pi > np.min(_angle_diff(yaw, yaw_origin)):
                return None
            if len(yaws) > 0:
                # 無矛盾な道路が存在する
                if angle_diff_th_in_degree / 180 * np.pi > np.min(_angle_diff(yaws, yaw)):
                    return yaw

    if len(yaws) > 0:
        # どの道路とも {angle_diff_th_in_degree}度 以上ずれてる場合は修正
        if angle_diff_th_in_degree / 180 * np.pi <= np.min(_angle_diff(yaws, yaw_origin)):
            min_yaw = yaws[np.argmin(np.stack([
                _angle_diff(yaws, yaw_origin),
                # _angle_diff(yaws, yaw_origin + np.pi / 2),
                _angle_diff(yaws, yaw_origin - np.pi),
                # _angle_diff(yaws, yaw_origin - np.pi / 2),
            ], axis=-1).min(axis=-1))]
            return min_yaw
    return None


def _try_to_fix_agent_yaw(
    sat_rast: SemanticRasterizer,
    agent: np.ndarray,
    selected_track_id: int,
    history_frames: np.ndarray,
    history_agents: List[np.ndarray],
    history_num_frames: int,
    angle_diff_th_in_degree: float,
    distance_th_in_agent: float,
    num_use_history: int,
    lane_polygon_buffer: float,
) -> None:

    if PERCEPTION_LABELS[np.argmax(agent["label_probabilities"])] != "PERCEPTION_LABEL_CAR":
        return agent, False

    agent_centroid_m = agent["centroid"]
    agent_yaw_rad = float(agent["yaw"])

    # TODO(kaizaburo): history_positions も world coords で計算する (それに合わせてハイパラも調整する)
    world_from_agent = compute_agent_pose(agent_centroid_m, agent_yaw_rad)
    agent_from_world = np.linalg.inv(world_from_agent)

    # history_num_frames + 1 because it also includes the current frame
    history_coords_offset, _, history_availability = _create_targets_for_deep_prediction(
        history_num_frames + 1, history_frames, selected_track_id, history_agents, agent_from_world, agent_yaw_rad
    )

    history_positions = []
    for pos, avail in zip(history_coords_offset, history_availability):
        if avail == 0.0:
            break
        history_positions.append(pos)
    history_positions = np.array(history_positions)

    new_yaw = fix_yaw_by_surrounding_lane_and_history(
        sat_rast=sat_rast,
        centroid=agent_centroid_m,
        yaw_origin=agent_yaw_rad,
        history_positions=history_positions,
        angle_diff_th_in_degree=angle_diff_th_in_degree,
        distance_th_in_agent=distance_th_in_agent,
        num_use_history=num_use_history,
        lane_polygon_buffer=lane_polygon_buffer,
    )
    if new_yaw is not None:
        agent = agent.copy()
        agent["yaw"] = new_yaw
        return agent, True
    else:
        return agent, False


def generate_agent_sample_fixing_yaw(
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
    sat_rast: SemanticRasterizer,
    rasterizer: Optional[Rasterizer] = None,
    perturbation: Optional[Perturbation] = None,
    angle_diff_th_in_degree: float = 60,
    distance_th_in_agent: float = 0.5,
    num_use_history: int = 2,
    lane_polygon_buffer: float = 1.5,
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

    tl_slice = get_tl_faces_slice_from_frames(history_frames[-1], history_frames[0])  # -1 is the farthest
    # sync interval with the traffic light faces array
    history_frames["traffic_light_faces_index_interval"] -= tl_slice.start
    history_tl_faces = filter_tl_faces_by_frames(history_frames, tl_faces[tl_slice].copy())

    if perturbation is not None:
        history_frames, future_frames = perturbation.perturb(
            history_frames=history_frames, future_frames=future_frames
        )

    # State you want to predict the future of.
    cur_frame = history_frames[0]
    cur_agents = history_agents[0]

    if selected_track_id is None:
        agent_centroid_m = cur_frame["ego_translation"][:2]
        agent_yaw_rad = rotation33_as_yaw(cur_frame["ego_rotation"])
        agent_extent_m = np.asarray((EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, EGO_EXTENT_HEIGHT))
        selected_agent = None
        yaw_fixed = False
    else:
        # this will raise IndexError if the agent is not in the frame or under agent-threshold
        # this is a strict error, we cannot recover from this situation
        try:
            agent = filter_agents_by_track_id(
                filter_agents_by_labels(cur_agents, filter_agents_threshold), selected_track_id
            )[0]
        except IndexError:
            raise ValueError(f" track_id {selected_track_id} not in frame or below threshold")

        # Try to fix agent's yaw
        agent, yaw_fixed = _try_to_fix_agent_yaw(
            sat_rast=sat_rast,
            agent=agent,
            selected_track_id=selected_track_id,
            history_frames=history_frames,
            history_agents=history_agents,
            history_num_frames=history_num_frames,
            angle_diff_th_in_degree=angle_diff_th_in_degree,
            distance_th_in_agent=distance_th_in_agent,
            num_use_history=num_use_history,
            lane_polygon_buffer=lane_polygon_buffer,
        )

        agent_centroid_m = agent["centroid"]
        agent_yaw_rad = float(agent["yaw"])
        agent_extent_m = agent["extent"]
        selected_agent = agent

    input_im = (
        None
        if not rasterizer
        else rasterizer.rasterize(history_frames, history_agents, history_tl_faces, selected_agent)
    )

    world_from_agent = compute_agent_pose(agent_centroid_m, agent_yaw_rad)
    agent_from_world = np.linalg.inv(world_from_agent)
    raster_from_world = render_context.raster_from_world(agent_centroid_m, agent_yaw_rad)

    future_coords_offset, future_yaws_offset, future_availability = _create_targets_for_deep_prediction(
        future_num_frames, future_frames, selected_track_id, future_agents, agent_from_world, agent_yaw_rad
    )

    # history_num_frames + 1 because it also includes the current frame
    history_coords_offset, history_yaws_offset, history_availability = _create_targets_for_deep_prediction(
        history_num_frames + 1, history_frames, selected_track_id, history_agents, agent_from_world, agent_yaw_rad
    )

    return {
        "image": input_im,
        "target_positions": future_coords_offset,
        "target_yaws": future_yaws_offset,
        "target_availabilities": future_availability,
        "history_positions": history_coords_offset,
        "history_yaws": history_yaws_offset,
        "history_availabilities": history_availability,
        "world_to_image": raster_from_world,  # TODO deprecate
        "raster_from_agent": raster_from_world @ world_from_agent,
        "raster_from_world": raster_from_world,
        "agent_from_world": agent_from_world,
        "world_from_agent": world_from_agent,
        "centroid": agent_centroid_m,
        "yaw": agent_yaw_rad,
        "extent": agent_extent_m,
        "yaw_fixed": yaw_fixed,
    }
