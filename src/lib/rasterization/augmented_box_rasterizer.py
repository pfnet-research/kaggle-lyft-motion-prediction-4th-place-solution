from typing import List, Optional

import numpy as np

from l5kit.data.filter import filter_agents_by_labels, filter_agents_by_track_id
from l5kit.geometry import rotation33_as_yaw
from l5kit.rasterization.render_context import RenderContext
from l5kit.rasterization.box_rasterizer import get_ego_as_agent, draw_boxes, BoxRasterizer


class AugmentedBoxRasterizer(BoxRasterizer):
    @staticmethod
    def from_cfg(cfg, data_manager=None, eval=False):
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
        return AugmentedBoxRasterizer(
            render_context,
            filter_agents_threshold,
            history_num_frames,
            raster_cfg.get("agent_drop_ratio", 0.9),
            raster_cfg.get("agent_drop_prob", -1.0),
            raster_cfg.get("min_extent_ratio", 0.8),
            raster_cfg.get("max_extent_ratio", 1.2),
            eval=eval
        )

    def __init__(
        self,
        render_context: RenderContext,
        filter_agents_threshold: float,
        history_num_frames: int,
        agent_drop_ratio: float = 0.9,
        agent_drop_prob: float = -1.0,
        min_extent_ratio: float = 0.8,
        max_extent_ratio: float = 1.2,
        eval: bool = False,
    ):
        """

        Args:
            render_context (RenderContext): Render context
            filter_agents_threshold (float): Value between 0 and 1 used to filter uncertain agent detections
            history_num_frames (int): Number of frames to rasterise in the past
        """
        super(AugmentedBoxRasterizer, self).__init__(render_context, filter_agents_threshold, history_num_frames)
        # --- These are called inside super init ---
        # self.render_context = render_context
        # self.raster_size = render_context.raster_size_px
        # self.filter_agents_threshold = filter_agents_threshold
        # self.history_num_frames = history_num_frames

        self.raster_channels = (self.history_num_frames + 1) * 5
        self.agent_drop_ratio = agent_drop_ratio
        self.agent_drop_prob = agent_drop_prob
        self.min_extent_ratio = min_extent_ratio
        self.max_extent_ratio = max_extent_ratio
        self.eval = eval  # Evaluation mode, No augmentation is applied when `True`.
        if eval:
            print("AugmentedBoxRasterizer eval mode is True!")

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
        agents_images = np.zeros(out_shape, dtype=np.uint8)
        ego_images = np.zeros(out_shape, dtype=np.uint8)

        # --- 1. prepare agent keep indices for random agent drop augmentation ---
        track_ids = np.concatenate([a["track_id"] for a in history_agents])
        unique_track_ids = np.unique(track_ids).astype(np.int64)
        n_max_agents = int(np.max(unique_track_ids) + 1)  # +1 for host car.

        unique_track_ids = np.concatenate([[0], unique_track_ids])  # Add Host car, with id=0.
        n_unique_agents = len(unique_track_ids)
        # if not np.all(unique_track_ids == np.arange(np.max(unique_track_ids) + 1)):
        #     # It occured!! --> unique_track_ids is not continuous. Some numbers are filtered out.
        #     print("unique_track_ids", unique_track_ids, "is not continuous!!!")
        if not self.eval and np.random.uniform() < self.agent_drop_ratio:
            if self.agent_drop_prob < 0:
                # Randomly decide number of agents to drop.
                # 0 represents host car.
                n_keep_agents = np.random.randint(0, n_unique_agents)
                agent_keep_indices = np.random.choice(unique_track_ids, n_keep_agents, replace=False)
            else:
                # Decide agents to drop or not by agent_drop_prob.
                agent_keep_indices = unique_track_ids[
                    np.random.uniform(0.0, 1.0, (n_unique_agents,)) > self.agent_drop_prob]
                n_keep_agents = len(agent_keep_indices)
            # Must keep ego agent!
            if agent["track_id"] not in agent_keep_indices:
                agent_keep_indices = np.append(agent_keep_indices, agent["track_id"])
        else:
            n_keep_agents = n_unique_agents
            # keep all agents
            agent_keep_indices = None

        # --- 2. prepare extent scale augmentation ratio ----
        # TODO: create enough number of extent_ratio array. Actually n_keep_agents suffice but create n_max_agents
        # for simplicity..
        if self.eval:
            # No augmentation.
            agents_extent_ratio = np.ones((n_max_agents, 3))
        elif self.min_extent_ratio == self.max_extent_ratio:
            agents_extent_ratio = np.ones((n_max_agents, 3)) * self.min_extent_ratio
        else:
            agents_extent_ratio = np.random.uniform(self.min_extent_ratio, self.max_extent_ratio, (n_max_agents, 3))
        ego_extent_ratio = agents_extent_ratio[0]

        for i, (frame, agents_) in enumerate(zip(history_frames, history_agents)):
            agents = filter_agents_by_labels(agents_, self.filter_agents_threshold)
            if agent_keep_indices is not None:
                # --- 1. apply agent drop augmentation ---
                agents = agents[np.isin(agents["track_id"], agent_keep_indices)]
            # note the cast is for legacy support of dataset before April 2020
            av_agent = get_ego_as_agent(frame).astype(agents.dtype)
            # 2. --- apply extent scale augmentation ---
            # TODO: Need to convert agents["track_id"] --> index based on `agent_keep_indices`,
            # if we only create `agents_extent_ratio` of size `n_keep_agents`.
            agents["extent"] *= agents_extent_ratio[agents["track_id"]]
            av_agent[0]["extent"] *= ego_extent_ratio

            if agent is None:
                agents_image = draw_boxes(self.raster_size, raster_from_world, agents, 255)
                ego_image = draw_boxes(self.raster_size, raster_from_world, av_agent, 255)
            else:
                agent_ego = filter_agents_by_track_id(agents, agent["track_id"])
                if agent_keep_indices is None or 0 in agent_keep_indices:
                    agents = np.append(agents, av_agent)
                if len(agent_ego) == 0:  # agent not in this history frame
                    agents_image = draw_boxes(self.raster_size, raster_from_world, agents, 255)
                    ego_image = np.zeros_like(agents_image)
                else:  # add av to agents and remove the agent from agents
                    agents = agents[agents != agent_ego[0]]
                    agents_image = draw_boxes(self.raster_size, raster_from_world, agents, 255)
                    ego_image = draw_boxes(self.raster_size, raster_from_world, agent_ego, 255)

            agents_images[..., i] = agents_image
            ego_images[..., i] = ego_image

        # combine such that the image consists of [agent_t, agent_t-1, agent_t-2, ego_t, ego_t-1, ego_t-2]
        out_im = np.concatenate((agents_images, ego_images), -1)

        return out_im.astype(np.float32) / 255
