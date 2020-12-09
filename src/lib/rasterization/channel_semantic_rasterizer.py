from collections import defaultdict
from typing import List, Optional, Tuple, cast

import cv2
import numpy as np
from l5kit.data import MapAPI, filter_tl_faces_by_status
from l5kit.geometry import rotation33_as_yaw, transform_point, transform_points
from l5kit.rasterization import SemanticRasterizer, RenderContext
from l5kit.rasterization.rasterizer_builder import _load_metadata, get_hardcoded_world_to_ecef
from l5kit.rasterization.semantic_rasterizer import CV2_SHIFT, cv2_subpixel, elements_within_bounds


class ChannelSemanticRasterizer(SemanticRasterizer):
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

        return ChannelSemanticRasterizer(render_context, semantic_map_filepath, world_to_ecef)

    def __init__(
        self,
        render_context: RenderContext, semantic_map_path: str, world_to_ecef: np.ndarray,
    ):
        super(ChannelSemanticRasterizer, self).__init__(render_context, semantic_map_path, world_to_ecef)
        self.raster_channels = 6

    def rasterize(
        self,
        history_frames: np.ndarray,
        history_agents: List[np.ndarray],
        history_tl_faces: List[np.ndarray],
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

        sem_im = self.render_semantic_map(center_in_world_m, raster_from_world, history_tl_faces[0])
        return sem_im.astype(np.float32) / 255

    def render_semantic_map(
        self, center_world: np.ndarray, raster_from_world: np.ndarray, tl_faces: np.ndarray
    ) -> np.ndarray:
        """Renders the semantic map at given x,y coordinates.

        Args:
            center_world (np.ndarray): XY of the image center in world ref system
            raster_from_world (np.ndarray):
            tl_faces (np.ndarray):

        Returns:
            np.ndarray: RGB raster

        """

        # img = 255 * np.ones(shape=(self.raster_size[1], self.raster_size[0], 3), dtype=np.uint8)
        img = np.zeros(shape=(self.raster_channels, self.raster_size[1], self.raster_size[0]), dtype=np.uint8)

        # filter using half a radius from the center
        raster_radius = float(np.linalg.norm(self.raster_size * self.pixel_size)) / 2

        # get active traffic light faces
        active_tl_ids = set(filter_tl_faces_by_status(tl_faces, "ACTIVE")["face_id"].tolist())

        # plot lanes
        lanes_lines = defaultdict(list)

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
            cv2.fillPoly(img[0], [lanes_area], 255, lineType=cv2.LINE_AA, shift=CV2_SHIFT)

            lane_type = "default"  # no traffic light face is controlling this lane
            lane_tl_ids = set([MapAPI.id_as_str(la_tc) for la_tc in lane.traffic_controls])
            for tl_id in lane_tl_ids.intersection(active_tl_ids):
                if self.proto_API.is_traffic_face_colour(tl_id, "red"):
                    lane_type = "red"
                elif self.proto_API.is_traffic_face_colour(tl_id, "green"):
                    lane_type = "green"
                elif self.proto_API.is_traffic_face_colour(tl_id, "yellow"):
                    lane_type = "yellow"

            lanes_lines[lane_type].extend([xy_left, xy_right])

        # --- Traffic lights ---
        # cv2.polylines(img, lanes_lines["default"], False, (255, 217, 82), lineType=cv2.LINE_AA, shift=CV2_SHIFT)
        # cv2.polylines(img, lanes_lines["green"], False, (0, 255, 0), lineType=cv2.LINE_AA, shift=CV2_SHIFT)
        # cv2.polylines(img, lanes_lines["yellow"], False, (255, 255, 0), lineType=cv2.LINE_AA, shift=CV2_SHIFT)
        # cv2.polylines(img, lanes_lines["red"], False, (255, 0, 0), lineType=cv2.LINE_AA, shift=CV2_SHIFT)
        cv2.polylines(img[1], lanes_lines["default"], False, 255, lineType=cv2.LINE_AA, shift=CV2_SHIFT)
        cv2.polylines(img[2], lanes_lines["green"], False, 255, lineType=cv2.LINE_AA, shift=CV2_SHIFT)
        cv2.polylines(img[3], lanes_lines["yellow"], False, 255, lineType=cv2.LINE_AA, shift=CV2_SHIFT)
        cv2.polylines(img[4], lanes_lines["red"], False, 255, lineType=cv2.LINE_AA, shift=CV2_SHIFT)

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
