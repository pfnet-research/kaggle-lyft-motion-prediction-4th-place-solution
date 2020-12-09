from bisect import bisect
from collections import defaultdict
from itertools import chain
from typing import List, Optional, Tuple, cast

import cv2
import numpy as np
from l5kit.data import MapAPI, filter_tl_faces_by_status
from l5kit.geometry import rotation33_as_yaw, transform_point, transform_points
from l5kit.rasterization import RenderContext, SemanticRasterizer
from l5kit.rasterization.rasterizer_builder import _load_metadata, get_hardcoded_world_to_ecef
from l5kit.rasterization.semantic_rasterizer import CV2_SHIFT, cv2_subpixel, elements_within_bounds


def _active_exists(tl_faces: np.ndarray, tl_id: str):
    if len(tl_faces) == 0:
        return False
    tl_faces = tl_faces[tl_faces["face_id"] == tl_id]
    if len(tl_faces) == 0:
        return False
    return np.any(tl_faces["traffic_light_face_status"][:, 0])


# Functionality not tested!
# def active_duration_binary(all_history_tl_faces: List[np.ndarray], tl_id: str) -> int:
#     # Binary search.
#     history_len = len(all_history_tl_faces)
#     lo, hi = 0, history_len
#     while lo < hi:
#         mid = (lo+hi)//2
#         if _active_exists(all_history_tl_faces[mid], tl_id):
#             hi = mid
#         else:
#             lo = mid+1
#     return lo


def active_duration(all_history_tl_faces: List[np.ndarray], tl_id: str) -> int:
    # Brute force search from earliest
    history_len = len(all_history_tl_faces)
    for i in range(history_len)[::-1]:
        if _active_exists(all_history_tl_faces[i], tl_id):
            return i
    raise ValueError("active tl_faces not found!!")


class ChannelSemanticTLRasterizer(SemanticRasterizer):
    @staticmethod
    def from_cfg(cfg, data_manager):
        raster_cfg = cfg["raster_params"]
        # map_type = raster_cfg["map_type"]
        dataset_meta_key = raster_cfg["dataset_meta_key"]

        render_context = RenderContext(
            raster_size_px=np.array(raster_cfg["raster_size"]),
            pixel_size_m=np.array(raster_cfg["pixel_size"]),
            center_in_raster_ratio=np.array(raster_cfg["ego_center"]),
        )
        # filter_agents_threshold = raster_cfg["filter_agents_threshold"]
        # history_num_frames = cfg["model_params"]["history_num_frames"]

        semantic_map_filepath = data_manager.require(raster_cfg["semantic_map_key"])
        try:
            dataset_meta = _load_metadata(dataset_meta_key, data_manager)
            world_to_ecef = np.array(dataset_meta["world_to_ecef"], dtype=np.float64)
        except (KeyError, FileNotFoundError):  # TODO remove when new dataset version is available
            world_to_ecef = get_hardcoded_world_to_ecef()

        lane_color_mode = raster_cfg.get("lane_color_mode", "normal")
        return ChannelSemanticTLRasterizer(
            render_context, semantic_map_filepath, world_to_ecef, lane_color_mode=lane_color_mode
        )

    def __init__(
        self, render_context: RenderContext, semantic_map_path: str, world_to_ecef: np.ndarray,
        lane_color_mode: str = "normal",
    ):
        super(ChannelSemanticTLRasterizer, self).__init__(render_context, semantic_map_path, world_to_ecef)
        self.raster_channels = 6
        self.lane_color_mode = lane_color_mode  # "normal", "inverse" or "none".

    def rasterize(
        self,
        history_frames: np.ndarray,
        history_agents: List[np.ndarray],
        all_history_tl_faces: List[np.ndarray],
        agent: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if agent is None:
            ego_translation_m = history_frames[0]["ego_translation"]
            ego_yaw_rad = rotation33_as_yaw(history_frames[0]["ego_rotation"])
        else:
            ego_translation_m = np.append(agent["centroid"], history_frames[0]["ego_translation"][-1])
            ego_yaw_rad = agent["yaw"]

        raster_from_world = self.render_context.raster_from_world(ego_translation_m, ego_yaw_rad)
        world_from_raster = np.linalg.inv(raster_from_world)

        # get XY of center pixel in world coordinates
        center_in_raster_px = np.asarray(self.raster_size) * (0.5, 0.5)
        center_in_world_m = transform_point(center_in_raster_px, world_from_raster)

        sem_im = self.render_semantic_map(center_in_world_m, raster_from_world, all_history_tl_faces)
        return sem_im.astype(np.float32) / 255

    def render_semantic_map(
        self, center_world: np.ndarray, raster_from_world: np.ndarray, all_history_tl_faces: List[np.ndarray]
    ) -> np.ndarray:
        """Renders the semantic map at given x,y coordinates.

        Args:
            center_world (np.ndarray): XY of the image center in world ref system
            raster_from_world (np.ndarray):
            all_history_tl_faces (list of np.ndarray):

        Returns:
            np.ndarray: RGB raster

        """
        tl_faces = all_history_tl_faces[0]

        # img = 255 * np.ones(shape=(self.raster_size[1], self.raster_size[0], 3), dtype=np.uint8)
        img = np.zeros(shape=(self.raster_channels, self.raster_size[1], self.raster_size[0]), dtype=np.uint8)

        # filter using half a radius from the center
        raster_radius = float(np.linalg.norm(self.raster_size * self.pixel_size)) / 2

        # get active traffic light faces
        active_tl_ids = set(filter_tl_faces_by_status(tl_faces, "ACTIVE")["face_id"].tolist())

        # plot lanes
        lanes_lines = defaultdict(list)
        lanes_active_frames = defaultdict(list)

        for idx in elements_within_bounds(center_world, self.bounds_info["lanes"]["bounds"], raster_radius):
            lane = self.proto_API[self.bounds_info["lanes"]["ids"][idx]].element.lane

            # get image coords
            lane_coords = self.proto_API.get_lane_coords(self.bounds_info["lanes"]["ids"][idx])
            xy_left = cv2_subpixel(transform_points(lane_coords["xyz_left"][:, :2], raster_from_world))
            xy_right = cv2_subpixel(transform_points(lane_coords["xyz_right"][:, :2], raster_from_world))
            lanes_area = np.vstack((xy_left, np.flip(xy_right, 0)))  # start->end left then end->start right

            # --- lanes ---
            # Note(lberg): this called on all polygons skips some of them, don't know why
            # cv2.fillPoly(img, [lanes_area], (17, 17, 31), lineType=cv2.LINE_AA, shift=CV2_SHIFT)
            if self.lane_color_mode == "normal":
                # lane is 255, outside is 0
                cv2.fillPoly(img[0], [lanes_area], 255, lineType=cv2.LINE_AA, shift=CV2_SHIFT)
            elif self.lane_color_mode == "inverse":
                # lane is 0, outside is 255
                cv2.fillPoly(img[0], [lanes_area], 255, lineType=cv2.LINE_AA, shift=CV2_SHIFT)
                img[0] = 255 - img[0]
            elif self.lane_color_mode == "none":
                # Do not draw lane...
                pass
            else:
                raise ValueError(f"[ERROR] Unexpected value self.lane_color_mode={self.lane_color_mode}")

            lane_type = "default"  # no traffic light face is controlling this lane
            lane_tl_ids = set([MapAPI.id_as_str(la_tc) for la_tc in lane.traffic_controls])
            for tl_id in lane_tl_ids.intersection(active_tl_ids):
                n_active_tl = 0
                # This is not exclusive, use all information...
                if self.proto_API.is_traffic_face_colour(tl_id, "red"):
                    lane_type = "red"
                    lanes_lines[lane_type].append([xy_left, xy_right])
                    active_frames = active_duration(all_history_tl_faces, tl_id)
                    lanes_active_frames[lane_type].append(active_frames)
                    n_active_tl += 1
                if self.proto_API.is_traffic_face_colour(tl_id, "green"):
                    lane_type = "green"
                    lanes_lines[lane_type].append([xy_left, xy_right])
                    active_frames = active_duration(all_history_tl_faces, tl_id)
                    lanes_active_frames[lane_type].append(active_frames)
                    n_active_tl += 1
                if self.proto_API.is_traffic_face_colour(tl_id, "yellow"):
                    lane_type = "yellow"
                    lanes_lines[lane_type].append([xy_left, xy_right])
                    active_frames = active_duration(all_history_tl_faces, tl_id)
                    lanes_active_frames[lane_type].append(active_frames)
                    n_active_tl += 1
                if n_active_tl > 1:
                    print("[DEBUG] n_active_tl", n_active_tl)

            if lane_type == "default":
                lanes_lines[lane_type].extend([xy_left, xy_right])

        # --- Traffic lights ---
        # import IPython; IPython.embed()
        # cv2.polylines(img, lanes_lines["default"], False, (255, 217, 82), lineType=cv2.LINE_AA, shift=CV2_SHIFT)
        # cv2.polylines(img, lanes_lines["green"], False, (0, 255, 0), lineType=cv2.LINE_AA, shift=CV2_SHIFT)
        # cv2.polylines(img, lanes_lines["yellow"], False, (255, 255, 0), lineType=cv2.LINE_AA, shift=CV2_SHIFT)
        # cv2.polylines(img, lanes_lines["red"], False, (255, 0, 0), lineType=cv2.LINE_AA, shift=CV2_SHIFT)

        # print("[DEBUG] lanes_active_frames", lanes_active_frames)
        cv2.polylines(img[1], lanes_lines["default"], False, 255, lineType=cv2.LINE_AA, shift=CV2_SHIFT)
        for ch, color in zip([2, 3, 4], ["green", "yellow", "red"]):
            lines = lanes_lines[color]
            active_frames = lanes_active_frames[color]
            if len(active_frames) == 0:
                continue
            # if np.all(np.array(active_frames) == active_frames[0]):
            #     # draw all lines at once, same color.
            #     color = 255 - active_frames[0]  # active_frames 0~250
            #     cv2.polylines(
            #         img[ch], list(chain.from_iterable(lines)), False, color, lineType=cv2.LINE_AA, shift=CV2_SHIFT
            #     )
            # else:
            # draw separately
            for i in range(len(lines)):
                color = 255 - active_frames[i]  # active_frames 0~250
                cv2.polylines(img[ch], lines[i], False, color, lineType=cv2.LINE_AA, shift=CV2_SHIFT)
        # cv2.polylines(img[2], lanes_lines["green"], False, 255, lineType=cv2.LINE_AA, shift=CV2_SHIFT)
        # cv2.polylines(img[3], lanes_lines["yellow"], False, 255, lineType=cv2.LINE_AA, shift=CV2_SHIFT)
        # cv2.polylines(img[4], lanes_lines["red"], False, 255, lineType=cv2.LINE_AA, shift=CV2_SHIFT)

        # plot crosswalks
        crosswalks = []
        for idx in elements_within_bounds(center_world, self.bounds_info["crosswalks"]["bounds"], raster_radius):
            crosswalk = self.proto_API.get_crosswalk_coords(self.bounds_info["crosswalks"]["ids"][idx])

            xy_cross = cv2_subpixel(transform_points(crosswalk["xyz"][:, :2], raster_from_world))
            crosswalks.append(xy_cross)

        # --- Cross Walks ---
        # cv2.polylines(img, crosswalks, True, (255, 117, 69), lineType=cv2.LINE_AA, shift=CV2_SHIFT)
        cv2.polylines(img[5], crosswalks, True, 255, lineType=cv2.LINE_AA, shift=CV2_SHIFT)
        # ch, h, w --> h, w, ch
        img = img.transpose((1, 2, 0))
        return img

    def to_rgb(self, in_im: np.ndarray, **kwargs: dict) -> np.ndarray:
        # return (in_im * 255).astype(np.uint8)
        raise NotImplementedError
