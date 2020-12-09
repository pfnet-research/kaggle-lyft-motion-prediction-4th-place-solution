from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from l5kit.data import PERCEPTION_LABEL_TO_INDEX

from l5kit.data.zarr_dataset import AGENT_DTYPE

from l5kit.data.filter import filter_agents_by_labels, filter_agents_by_track_id
from l5kit.geometry import rotation33_as_yaw, transform_points
from l5kit.geometry.transform import yaw_as_rotation33
from l5kit.rasterization.rasterizer import EGO_EXTENT_HEIGHT, EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, Rasterizer
from l5kit.rasterization.render_context import RenderContext
from l5kit.rasterization.semantic_rasterizer import CV2_SHIFT, cv2_subpixel
from l5kit.rasterization.box_rasterizer import get_ego_as_agent, draw_boxes


UNKNOWN_LABEL_INDEX = PERCEPTION_LABEL_TO_INDEX["PERCEPTION_LABEL_UNKNOWN"]
CAR_LABEL_INDEX = PERCEPTION_LABEL_TO_INDEX["PERCEPTION_LABEL_CAR"]
CYCLIST_LABEL_INDEX = PERCEPTION_LABEL_TO_INDEX["PERCEPTION_LABEL_CYCLIST"]
PEDESTRIAN_LABEL_INDEX = PERCEPTION_LABEL_TO_INDEX["PERCEPTION_LABEL_PEDESTRIAN"]


class AgentTypeBoxRasterizer(Rasterizer):
    @staticmethod
    def from_cfg(cfg, data_manager=None):
        raster_cfg = cfg["raster_params"]
        # map_type = raster_cfg["map_type"]
        # dataset_meta_key = raster_cfg["dataset_meta_key"]

        render_context = RenderContext(
            raster_size_px=np.array(raster_cfg["raster_size"]),
            pixel_size_m=np.array(raster_cfg["pixel_size"]),
            center_in_raster_ratio=np.array(raster_cfg["ego_center"]),
        )

        enable_selected_agent_channels = raster_cfg.get("enable_selected_agent_channels", False)

        # raster_size: Tuple[int, int] = cast(Tuple[int, int], tuple(raster_cfg["raster_size"]))
        # pixel_size = np.array(raster_cfg["pixel_size"])
        # ego_center = np.array(raster_cfg["ego_center"])
        filter_agents_threshold = raster_cfg["filter_agents_threshold"]
        history_num_frames = cfg["model_params"]["history_num_frames"]
        return AgentTypeBoxRasterizer(
            render_context, filter_agents_threshold, history_num_frames, enable_selected_agent_channels
        )

    def __init__(
        self,
        render_context: RenderContext,
        filter_agents_threshold: float,
        history_num_frames: int,
        enable_selected_agent_channels: bool,
    ):
        """

        Args:
            render_context (RenderContext): Render context
            filter_agents_threshold (float): Value between 0 and 1 used to filter uncertain agent detections
            history_num_frames (int): Number of frames to rasterise in the past
            enable_selected_agent_channels (bool): Whether to add channels only for the selected agent
        """
        super(AgentTypeBoxRasterizer, self).__init__()
        self.render_context = render_context
        self.raster_size = render_context.raster_size_px
        self.filter_agents_threshold = filter_agents_threshold
        self.history_num_frames = history_num_frames
        self.enable_selected_agent_channels = enable_selected_agent_channels

        if enable_selected_agent_channels:
            self.raster_channels = (self.history_num_frames + 1) * 6
        else:
            self.raster_channels = (self.history_num_frames + 1) * 5

    def rasterize(
        self,
        history_frames: np.ndarray,
        history_agents: List[np.ndarray],
        history_tl_faces: List[np.ndarray],
        agent: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # all frames are drawn relative to this one"
        frame = history_frames[0]
        if agent is None:
            ego_translation_m = history_frames[0]["ego_translation"]
            ego_yaw_rad = rotation33_as_yaw(frame["ego_rotation"])
        else:
            ego_translation_m = np.append(agent["centroid"], history_frames[0]["ego_translation"][-1])
            ego_yaw_rad = agent["yaw"]

        raster_from_world = self.render_context.raster_from_world(ego_translation_m, ego_yaw_rad)

        # this ensures we always end up with fixed size arrays, +1 is because current time is also in the history
        out_shape = (self.raster_size[1], self.raster_size[0], self.history_num_frames + 1)
        agents_unknown_images = np.zeros(out_shape, dtype=np.uint8)
        agents_car_images = np.zeros(out_shape, dtype=np.uint8)
        agents_cyclist_images = np.zeros(out_shape, dtype=np.uint8)
        agents_pedestrian_images = np.zeros(out_shape, dtype=np.uint8)
        ego_images = np.zeros(out_shape, dtype=np.uint8)

        if self.enable_selected_agent_channels:
            assert agent is not None
            selected_agent_images = np.zeros(out_shape, dtype=np.uint8)

        for i, (frame, agents) in enumerate(zip(history_frames, history_agents)):
            # TODO: filter_agents_threshold is not used.
            # Ignore filter_agents_threshold now
            threshold = 0.5
            agents_unknown = agents[agents["label_probabilities"][:, UNKNOWN_LABEL_INDEX] > threshold]
            agents_car = agents[agents["label_probabilities"][:, CAR_LABEL_INDEX] > threshold]
            agents_cyclist = agents[agents["label_probabilities"][:, CYCLIST_LABEL_INDEX] > threshold]
            agents_pedestrian = agents[agents["label_probabilities"][:, PEDESTRIAN_LABEL_INDEX] > threshold]
            # agents = filter_agents_by_labels(agents, self.filter_agents_threshold)

            # note the cast is for legacy support of dataset before April 2020
            agents_ego = get_ego_as_agent(frame).astype(agents.dtype)

            agents_unknown_image = draw_boxes(self.raster_size, raster_from_world, agents_unknown, 255)
            agents_car_image = draw_boxes(self.raster_size, raster_from_world, agents_car, 255)
            agents_cyclist_image = draw_boxes(self.raster_size, raster_from_world, agents_cyclist, 255)
            agents_pedestrian_image = draw_boxes(self.raster_size, raster_from_world, agents_pedestrian, 255)
            ego_image = draw_boxes(self.raster_size, raster_from_world, agents_ego, 255)

            agents_unknown_images[..., i] = agents_unknown_image
            agents_car_images[..., i] = agents_car_image
            agents_cyclist_images[..., i] = agents_cyclist_image
            agents_pedestrian_images[..., i] = agents_pedestrian_image
            ego_images[..., i] = ego_image

            if self.enable_selected_agent_channels:
                assert agent is not None
                selected_agent = filter_agents_by_track_id(agents, agent["track_id"])
                if len(selected_agent) == 0:  # agent not in this history frame
                    selected_agent_image = np.zeros_like(ego_image)
                else:  # add av to agents and remove the agent from agents
                    selected_agent_image = draw_boxes(self.raster_size, raster_from_world, selected_agent, 255)
                selected_agent_images[..., i] = selected_agent_image

        images = [
            agents_unknown_images,
            agents_car_images,
            agents_cyclist_images,
            agents_pedestrian_images,
            ego_images
        ]
        if self.enable_selected_agent_channels:
            images.append(selected_agent_images)

        out_im = np.concatenate(images, axis=-1)

        assert out_im.shape[-1] == self.raster_channels
        return out_im.astype(np.float32) / 255

    def to_rgb(self, in_im: np.ndarray, **kwargs: dict) -> np.ndarray:
        """
        get an rgb image where agents further in the past have faded colors

        Args:
            in_im: the output of the rasterize function
            kwargs: this can be used for additional customization (such as colors)

        Returns: an RGB image with agents and ego coloured with fading colors
        """
        hist_frames = self.history_num_frames + 1
        in_im = np.transpose(in_im, (2, 0, 1))
        assert in_im.shape[0] == self.raster_channels

        # this is similar to the draw history code

        out_im = np.zeros((self.raster_size[1], self.raster_size[0], 3), dtype=np.float32)
        agent_colors = [
            (0.5, 0.5, 0.5),  # gray for unknown
            (0, 0, 1),  # blue for car
            (1, 1, 0),  # yellow for cyclist
            (1, 0, 0),  # red for pedestrian
            (0, 1, 0)  # green for ego
        ]
        if self.enable_selected_agent_channels:
            agent_colors.append((0, 1, 1))  # aqua for selected

        for i in range(len(agent_colors)):
            out_im_agent = np.zeros((self.raster_size[1], self.raster_size[0], 3), dtype=np.float32)
            # unknown, car, cyclist, pedestrian, ego respectively.
            agent_chs = in_im[i*hist_frames:(i+1)*hist_frames][::-1]  # reverse to start from the furthest one
            agent_color = agent_colors[i]
            for ch in agent_chs:
                out_im_agent *= 0.85  # magic fading constant for the past
                out_im_agent[ch > 0] = agent_color
            out_im += out_im_agent

        out_im = (np.clip(out_im, 0, 1) * 255).astype(np.uint8)
        return out_im
