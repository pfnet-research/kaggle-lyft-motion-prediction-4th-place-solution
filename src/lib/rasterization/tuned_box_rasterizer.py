from typing import List, Optional, Tuple

import cv2
import numpy as np

from l5kit.data.zarr_dataset import AGENT_DTYPE
from l5kit.data.filter import filter_agents_by_labels, filter_agents_by_track_id
from l5kit.geometry import rotation33_as_yaw  # , transform_points
from l5kit.rasterization.rasterizer import EGO_EXTENT_HEIGHT, EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, Rasterizer
from l5kit.rasterization.render_context import RenderContext
from l5kit.rasterization.semantic_rasterizer import CV2_SHIFT, cv2_subpixel

from lib.utils.numba_utils import transform_points_nb


def get_ego_as_agent(frame: np.ndarray) -> np.ndarray:  # TODO this can be useful to have around
    """
    Get a valid agent with information from the frame AV. Ford Fusion extent is used

    Args:
        frame (np.ndarray): the frame we're interested in

    Returns: an agent np.ndarray of the AV

    """
    ego_agent = np.zeros(1, dtype=AGENT_DTYPE)
    ego_agent[0]["centroid"] = frame["ego_translation"][:2]
    ego_agent[0]["yaw"] = rotation33_as_yaw(frame["ego_rotation"])
    ego_agent[0]["extent"] = np.asarray((EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, EGO_EXTENT_HEIGHT))
    return ego_agent


def draw_boxes(
    raster_size: Tuple[int, int],
    raster_from_world: np.ndarray,
    agents: np.ndarray,
    color: int,
) -> np.ndarray:
    im = np.zeros((raster_size[1], raster_size[0]), dtype=np.uint8)

    corners_base_coords = np.asarray([[-1, -1], [-1, 1], [1, 1], [1, -1]])

    # corners = corners_base_coords[np.newaxis, :] agents["extent"][:, :2] / 2
    corners = corners_base_coords[np.newaxis, :, :] * agents["extent"][:, np.newaxis, :2] / 2

    # r_m_T = r_m.T;
    # r_m = [
    #     [cos(yaw), -sin(yaw)],
    #     [sin(yaw), cos(yaw)],
    # ]
    r_m_T = np.zeros((len(agents), 2, 2))
    r_m_T[:, 0, 0] = np.cos(agents["yaw"])
    r_m_T[:, 1, 1] = np.cos(agents["yaw"])
    r_m_T[:, 0, 1] = np.sin(agents["yaw"])
    r_m_T[:, 1, 0] = -np.sin(agents["yaw"])

    box_world_coords = np.sum(corners[:, :, :, np.newaxis] * r_m_T[:, np.newaxis, :, :], axis=-2)
    box_world_coords = box_world_coords + agents["centroid"][:, np.newaxis, :2]

    box_raster_coords = transform_points_nb(box_world_coords.reshape((-1, 2)), raster_from_world)

    # fillPoly wants polys in a sequence with points inside as (x,y)
    box_raster_coords = cv2_subpixel(box_raster_coords.reshape((-1, 4, 2)))
    cv2.fillPoly(im, box_raster_coords, color=color, lineType=cv2.LINE_AA, shift=CV2_SHIFT)
    return im


class TunedBoxRasterizer(Rasterizer):

    @staticmethod
    def from_cfg(cfg, data_manager=None):
        raster_cfg = cfg["raster_params"]

        render_context = RenderContext(
            raster_size_px=np.array(raster_cfg["raster_size"]),
            pixel_size_m=np.array(raster_cfg["pixel_size"]),
            center_in_raster_ratio=np.array(raster_cfg["ego_center"]),
        )

        filter_agents_threshold = raster_cfg["filter_agents_threshold"]
        history_num_frames = cfg["model_params"]["history_num_frames"]
        rotate_yaw = raster_cfg.get("rotate_yaw", True)
        return TunedBoxRasterizer(render_context, filter_agents_threshold, history_num_frames, rotate_yaw)

    def __init__(
        self, render_context: RenderContext, filter_agents_threshold: float, history_num_frames: int,
        rotate_yaw: bool = True,
    ):
        """

        Args:
            render_context (RenderContext): Render context
            filter_agents_threshold (float): Value between 0 and 1 used to filter uncertain agent detections
            history_num_frames (int): Number of frames to rasterise in the past
        """
        super(TunedBoxRasterizer, self).__init__()
        self.render_context = render_context
        self.raster_size = render_context.raster_size_px
        self.filter_agents_threshold = filter_agents_threshold
        self.history_num_frames = history_num_frames
        self.rotate_yaw = rotate_yaw

        self.raster_channels = (self.history_num_frames + 1) * 2

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
            ego_yaw_rad = rotation33_as_yaw(frame["ego_rotation"]) if self.rotate_yaw else 0.
        else:
            ego_translation_m = np.append(agent["centroid"], history_frames[0]["ego_translation"][-1])
            ego_yaw_rad = agent["yaw"] if self.rotate_yaw else 0.

        raster_from_world = self.render_context.raster_from_world(ego_translation_m, ego_yaw_rad)

        # this ensures we always end up with fixed size arrays, +1 is because current time is also in the history
        out_shape = (self.raster_size[1], self.raster_size[0], 2 * (self.history_num_frames + 1))
        ego_offset = (self.history_num_frames + 1)
        out_im = np.zeros(out_shape, dtype=np.uint8)

        for i, (frame, agents) in enumerate(zip(history_frames, history_agents)):
            agents = filter_agents_by_labels(agents, self.filter_agents_threshold)
            # note the cast is for legacy support of dataset before April 2020
            av_agent = get_ego_as_agent(frame).astype(agents.dtype)

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

            out_im[..., i] = agents_image
            out_im[..., ego_offset + i] = ego_image

        return out_im.astype(np.float32) / 255

    def to_rgb(self, in_im: np.ndarray, **kwargs: dict) -> np.ndarray:
        """
        get an rgb image where agents further in the past have faded colors

        Args:
            in_im: the output of the rasterize function
            kwargs: this can be used for additional customization (such as colors)

        Returns: an RGB image with agents and ego coloured with fading colors
        """
        hist_frames = in_im.shape[-1] // 2
        in_im = np.transpose(in_im, (2, 0, 1))

        # this is similar to the draw history code
        out_im_agent = np.zeros((self.raster_size[1], self.raster_size[0], 3), dtype=np.float32)
        agent_chs = in_im[:hist_frames][::-1]  # reverse to start from the furthest one
        agent_color = (0, 0, 1) if "agent_color" not in kwargs else kwargs["agent_color"]
        for ch in agent_chs:
            out_im_agent *= 0.85  # magic fading constant for the past
            out_im_agent[ch > 0] = agent_color

        out_im_ego = np.zeros((self.raster_size[1], self.raster_size[0], 3), dtype=np.float32)
        ego_chs = in_im[hist_frames:][::-1]
        ego_color = (0, 1, 0) if "ego_color" not in kwargs else kwargs["ego_color"]
        for ch in ego_chs:
            out_im_ego *= 0.85
            out_im_ego[ch > 0] = ego_color

        out_im = (np.clip(out_im_agent + out_im_ego, 0, 1) * 255).astype(np.uint8)
        return out_im
