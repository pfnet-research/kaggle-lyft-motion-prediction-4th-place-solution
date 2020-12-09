from typing import List, Optional, Tuple, Union, cast

import cv2
import numpy as np
from l5kit.data import filter_agents_by_labels
from l5kit.data.filter import filter_agents_by_track_id

from l5kit.geometry import rotation33_as_yaw, transform_points
from l5kit.rasterization import Rasterizer, RenderContext
from l5kit.rasterization.box_rasterizer import get_ego_as_agent, draw_boxes


def draw_velocity(
    raster_size: Tuple[int, int],
    world_to_image_space: np.ndarray,
    agents: np.ndarray,
    color: Union[int, Tuple[int, int, int]],
    velocity_scale: float = 1.0,
    thickness: int = 1,
    tipLength: float = 0.1,
) -> np.ndarray:
    """
    Draw multiple boxes in one sweep over the image.
    Boxes corners are extracted from agents, and the coordinates are projected in the image plane.
    Finally, cv2 draws the boxes.

    Args:
        raster_size (Tuple[int, int]): Desired output image size
        world_to_image_space (np.ndarray): 3x3 matrix to convert from world to image coordinated
        agents (np.ndarray): array of agents to be drawn
        color (Union[int, Tuple[int, int, int]]): single int or RGB color

        velocity_scale (float): how long velocity vector is drawn. bigger value will draw longer vector.
        thickness (int): thickness of arrow
        tipLength (float): arrow's tip length

    Returns:
        np.ndarray: the image with agents rendered. RGB if color RGB, otherwise GRAY
    """
    if isinstance(color, int):
        im = np.zeros(raster_size, dtype=np.uint8)
    else:
        im = np.zeros(raster_size + (3,), dtype=np.uint8)

    # (n_agents, 2)
    world_coords = agents["centroid"][:, :2]
    world_velocity = agents["velocity"]
    # print("world_coords", world_coords.shape, world_coords)
    # print("world_velocity", world_velocity.shape, world_velocity)
    image_coords = transform_points(world_coords, world_to_image_space)
    image_coords_next = transform_points(world_coords + world_velocity * velocity_scale, world_to_image_space)
    # image_velocity = transform_points(world_velocity, world_to_image_space)  # This does NOT work!
    # print("image_coords", image_coords.shape, image_coords)
    # print("image_velocity", image_velocity.shape, image_velocity)

    # fillPoly wants polys in a sequence with points inside as (x,y)
    pt1 = image_coords.astype(np.int64)
    pt2 = image_coords_next.astype(np.int64)
    # print("pt1", pt1.shape, pt1)
    # print("pt2", pt2.shape, pt2)

    for i in range(len(pt1)):
        cv2.arrowedLine(im, tuple(pt1[i]), tuple(pt2[i]), color, thickness=thickness, tipLength=tipLength)
    return im


class VelocityBoxRasterizer(Rasterizer):
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

        # raster_size: Tuple[int, int] = cast(Tuple[int, int], tuple(raster_cfg["raster_size"]))
        # pixel_size = np.array(raster_cfg["pixel_size"])
        # ego_center = np.array(raster_cfg["ego_center"])
        filter_agents_threshold = raster_cfg["filter_agents_threshold"]
        history_num_frames = cfg["model_params"]["history_num_frames"]
        return VelocityBoxRasterizer(render_context, filter_agents_threshold, history_num_frames)

    def __init__(
        self,
        render_context: RenderContext, filter_agents_threshold: float, history_num_frames: int,
    ):
        """

        Arguments:
            render_context:
                raster_size (Tuple[int, int]): Desired output image size
                pixel_size (np.ndarray): Dimensions of one pixel in the real world
                ego_center (np.ndarray): Center of ego in the image, [0.5,0.5] would be in the image center.
            filter_agents_threshold (float): Value between 0 and 1 used to filter uncertain agent detections
            history_num_frames (int): Number of frames to rasterise in the past
        """
        super(VelocityBoxRasterizer, self).__init__()
        self.render_context = render_context
        self.raster_size = render_context.raster_size_px
        self.pixel_size = render_context.pixel_size_m
        self.ego_center = render_context.center_in_raster_ratio
        self.filter_agents_threshold = filter_agents_threshold
        self.history_num_frames = history_num_frames
        self.raster_channels = (self.history_num_frames + 1) * 3

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
            ego_translation_m = frame["ego_translation"]
            ego_yaw_rad = rotation33_as_yaw(frame["ego_rotation"])
        else:
            ego_translation_m = np.append(agent["centroid"], history_frames[0]["ego_translation"][-1])
            ego_yaw_rad = agent["yaw"]

        if self.pixel_size[0] != self.pixel_size[1]:
            raise NotImplementedError("No support for non squared pixels yet")

        raster_from_world = self.render_context.raster_from_world(ego_translation_m, ego_yaw_rad)

        # this ensures we always end up with fixed size arrays, +1 is because current time is also in the history
        out_shape = (self.raster_size[1], self.raster_size[0], self.history_num_frames + 1)
        agents_images = np.zeros(out_shape, dtype=np.uint8)
        ego_images = np.zeros(out_shape, dtype=np.uint8)
        velocity_images = np.zeros(out_shape, dtype=np.uint8)

        for i, (frame, agents) in enumerate(zip(history_frames, history_agents)):
            agents = filter_agents_by_labels(agents, self.filter_agents_threshold)
            # note the cast is for legacy support of dataset before April 2020
            av_agent = get_ego_as_agent(frame).astype(agents.dtype)

            velocity_image = draw_velocity(self.raster_size, raster_from_world, np.append(agents, av_agent), 255)

            if agent is None:
                agents_image = draw_boxes(self.raster_size, raster_from_world, agents, 255)
                ego_image = draw_boxes(self.raster_size, raster_from_world, av_agent, 255)
            else:
                agent_ego = filter_agents_by_track_id(agents, agent["track_id"])
                if len(agent_ego) == 0:  # agent not in this history frame
                    agents_image = draw_boxes(self.raster_size, raster_from_world, np.append(agents, av_agent), 255)
                    ego_image = np.zeros_like(agents_image)
                else:  # add av to agents and remove the agent from agents
                    agents = agents[agents != agent_ego[0]]
                    agents_image = draw_boxes(self.raster_size, raster_from_world, np.append(agents, av_agent), 255)
                    ego_image = draw_boxes(self.raster_size, raster_from_world, agent_ego, 255)

            agents_images[..., i] = agents_image
            ego_images[..., i] = ego_image
            velocity_images[..., i] = velocity_image

        # combine such that the image consists of [agent_t, agent_t-1, agent_t-2, ego_t, ego_t-1, ego_t-2]
        out_im = np.concatenate((agents_images, ego_images, velocity_images), -1)

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
        hist_frames = in_im.shape[-1] // 3
        in_im = np.transpose(in_im, (2, 0, 1))

        # this is similar to the draw history code
        out_im_agent = np.zeros((self.raster_size[1], self.raster_size[0], 3), dtype=np.float32)
        agent_chs = in_im[:hist_frames][::-1]  # reverse to start from the furthest one
        agent_color = (0, 0, 1) if "agent_color" not in kwargs else kwargs["agent_color"]
        for ch in agent_chs:
            out_im_agent *= 0.85  # magic fading constant for the past
            out_im_agent[ch > 0] = agent_color

        out_im_ego = np.zeros((self.raster_size[1], self.raster_size[0], 3), dtype=np.float32)
        ego_chs = in_im[hist_frames:2*hist_frames][::-1]
        ego_color = (0, 1, 0) if "ego_color" not in kwargs else kwargs["ego_color"]
        for ch in ego_chs:
            out_im_ego *= 0.85
            out_im_ego[ch > 0] = ego_color

        out_im_vel = np.zeros((self.raster_size[1], self.raster_size[0], 3), dtype=np.float32)
        vel_chs = in_im[2*hist_frames:3*hist_frames][::-1]
        vel_color = (1, 0, 0) if "vel_color" not in kwargs else kwargs["ego_color"]
        for ch in vel_chs:
            out_im_vel *= 0.85
            out_im_vel[ch > 0] = vel_color

        out_im = (np.clip(out_im_agent + out_im_ego + out_im_vel, 0, 1) * 255).astype(np.uint8)
        print("out_im", out_im.shape)
        return out_im
