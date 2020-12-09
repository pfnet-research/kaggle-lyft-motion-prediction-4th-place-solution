from collections import defaultdict
from typing import List, Optional

import cv2
import numba as nb
import numpy as np

from l5kit.data.filter import filter_tl_faces_by_status
from l5kit.data.map_api import MapAPI
from l5kit.geometry import rotation33_as_yaw, transform_point, transform_points
from l5kit.rasterization.rasterizer import Rasterizer
from l5kit.rasterization.rasterizer_builder import _load_metadata, get_hardcoded_world_to_ecef
from l5kit.rasterization.render_context import RenderContext
# sub-pixel drawing precision constants
from tqdm import tqdm

from lib.data.tuned_map_api import TunedMapAPI
from lib.utils.numba_utils import transform_point_nb, transform_points_nb, elements_within_bounds_nb
from lib.utils.timer_utils import timer, timer_ms

CV2_SHIFT = 8  # how many bits to shift in drawing
CV2_SHIFT_VALUE = 2 ** CV2_SHIFT


@nb.jit(nb.int64[:, :](nb.float64[:, :], nb.int64), nopython=True, nogil=True)
def cv2_subpixel_nb(coords: np.ndarray, shift: int = CV2_SHIFT_VALUE) -> np.ndarray:
    """
    Cast coordinates to numpy.int but keep fractional part by previously multiplying by 2**CV2_SHIFT
    cv2 calls will use shift to restore original values with higher precision

    Args:
        coords (np.ndarray): XY coords as float

    Returns:
        np.ndarray: XY coords as int for cv2 shift draw
    """
    coords = coords * float(shift)
    coords_int = coords.astype(np.int64)
    return coords_int


@nb.jit(nb.int64[:, :](nb.float64[:, :], nb.float64[:, :], nb.int64), nopython=True, nogil=True)
def cv2_subpixel_transform_nb(
        points: np.ndarray, transf_matrix: np.ndarray, shift: int = CV2_SHIFT_VALUE
) -> np.ndarray:
    coords = transform_points_nb(points, transf_matrix)
    return cv2_subpixel_nb(coords, shift)


def cv2_subpixel_transform_nb_indices(
        indices, points: np.ndarray, transf_matrix: np.ndarray, shift: int = CV2_SHIFT_VALUE
) -> List:
    if len(indices) == 0:
        return []
    else:
        points_indices = points[indices]
        split_indices = np.cumsum([len(a) for a in points_indices])[:-1]
        concat_points = np.concatenate(points_indices, axis=0)
        concat_coords = transform_points_nb(concat_points, transf_matrix)
        concat_coords_int = cv2_subpixel_nb(concat_coords, shift)
        coords_int_list = np.split(concat_coords_int, split_indices)
        return coords_int_list


class TunedSemanticRasterizer(Rasterizer):
    """
    Rasteriser for the vectorised semantic map (generally loaded from json files).
    """

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

        rotate_yaw = raster_cfg.get("rotate_yaw", True)
        return TunedSemanticRasterizer(render_context, semantic_map_filepath, world_to_ecef, rotate_yaw)

    def __init__(
        self, render_context: RenderContext, semantic_map_path: str, world_to_ecef: np.ndarray, rotate_yaw: bool = True
    ):
        super(TunedSemanticRasterizer, self).__init__()
        self.render_context = render_context
        self.raster_size = render_context.raster_size_px
        self.pixel_size = render_context.pixel_size_m
        self.ego_center = render_context.center_in_raster_ratio

        self.world_to_ecef = world_to_ecef

        # self.proto_API = MapAPI(semantic_map_path, world_to_ecef)
        self.proto_API = TunedMapAPI(semantic_map_path, world_to_ecef)

        self.bounds_info = self.get_bounds()
        self.raster_channels = 3
        self.rotate_yaw = rotate_yaw

        # --- Other initialization for fast processing ---
        print("Init idx_to_traffic_control_id_list...")
        lane_ids = self.bounds_info["lanes"]["ids"]
        self.idx_to_traffic_control_id_list = [
            set([MapAPI.id_as_str(la_tc) for la_tc in self.proto_API[lane_ids[idx]].element.lane.traffic_controls])
            for idx in tqdm(range(len(lane_ids)))
        ]
        lane_coords_list = [
            self.proto_API.get_lane_coords(lane_ids[idx])
            for idx in tqdm(range(len(lane_ids)))
        ]
        # self.xyz_left_array = np.array([lane_coords["xyz_left"][:, :2] for lane_coords in lane_coords_list])
        # self.xyz_right_array = np.array([lane_coords["xyz_right"][:, :2] for lane_coords in lane_coords_list])

        self.xyz_array = np.array([
            np.concatenate([lane_coords["xyz_left"][:, :2], lane_coords["xyz_right"][::-1, :2]], axis=0)
            for lane_coords in lane_coords_list
        ], dtype=object)
        self.xyz_left_length = np.array([len(lane_coords["xyz_left"]) for lane_coords in lane_coords_list])

        bounds_cross_ids = self.bounds_info["crosswalks"]["ids"]
        self.crosswalk_2d_array = np.array([
            self.proto_API.get_crosswalk_coords(bounds_cross_ids[idx])["xyz"][:, :2]
            for idx in tqdm(range(len(bounds_cross_ids)))
        ], dtype=object)



    # TODO is this the right place for this function?
    def get_bounds(self) -> dict:
        """
        For each elements of interest returns bounds [[min_x, min_y],[max_x, max_y]] and proto ids
        Coords are computed by the MapAPI and, as such, are in the world ref system.

        Returns:
            dict: keys are classes of elements, values are dict with `bounds` and `ids` keys
        """
        lanes_ids = []
        crosswalks_ids = []

        # lanes_bounds = np.empty((0, 2, 2), dtype=np.float)  # [(X_MIN, Y_MIN), (X_MAX, Y_MAX)]
        # crosswalks_bounds = np.empty((0, 2, 2), dtype=np.float)  # [(X_MIN, Y_MIN), (X_MAX, Y_MAX)]

        lanes_bounds_list = []
        crosswalks_bounds_list = []

        for element in self.proto_API:
            element_id = MapAPI.id_as_str(element.id)

            if self.proto_API.is_lane(element):
                lane = self.proto_API.get_lane_coords(element_id)
                x_min = min(np.min(lane["xyz_left"][:, 0]), np.min(lane["xyz_right"][:, 0]))
                y_min = min(np.min(lane["xyz_left"][:, 1]), np.min(lane["xyz_right"][:, 1]))
                x_max = max(np.max(lane["xyz_left"][:, 0]), np.max(lane["xyz_right"][:, 0]))
                y_max = max(np.max(lane["xyz_left"][:, 1]), np.max(lane["xyz_right"][:, 1]))

                # lanes_bounds = np.append(lanes_bounds, np.asarray([[[x_min, y_min], [x_max, y_max]]]), axis=0)
                lanes_bounds_list.append(np.asarray([[[x_min, y_min], [x_max, y_max]]]))
                lanes_ids.append(element_id)

            if self.proto_API.is_crosswalk(element):
                crosswalk = self.proto_API.get_crosswalk_coords(element_id)
                x_min = np.min(crosswalk["xyz"][:, 0])
                y_min = np.min(crosswalk["xyz"][:, 1])
                x_max = np.max(crosswalk["xyz"][:, 0])
                y_max = np.max(crosswalk["xyz"][:, 1])

                # crosswalks_bounds = np.append(
                #     crosswalks_bounds, np.asarray([[[x_min, y_min], [x_max, y_max]]]), axis=0,
                # )
                crosswalks_bounds_list.append(np.asarray([[[x_min, y_min], [x_max, y_max]]]))
                crosswalks_ids.append(element_id)

        lanes_bounds = np.concatenate(lanes_bounds_list, axis=0)
        crosswalks_bounds = np.concatenate(crosswalks_bounds_list, axis=0)
        return {
            "lanes": {"bounds": lanes_bounds, "ids": lanes_ids},
            "crosswalks": {"bounds": crosswalks_bounds, "ids": crosswalks_ids},
        }

    def rasterize(
        self,
        history_frames: np.ndarray,
        history_agents: List[np.ndarray],
        history_tl_faces: List[np.ndarray],
        agent: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if agent is None:
            ego_translation_m = history_frames[0]["ego_translation"]
            ego_yaw_rad = rotation33_as_yaw(history_frames[0]["ego_rotation"]) if self.rotate_yaw else 0.
        else:
            ego_translation_m = np.append(agent["centroid"], history_frames[0]["ego_translation"][-1])
            ego_yaw_rad = agent["yaw"] if self.rotate_yaw else 0.

        raster_from_world = np.ascontiguousarray(self.render_context.raster_from_world(ego_translation_m, ego_yaw_rad))
        world_from_raster = np.linalg.inv(raster_from_world)

        # get XY of center pixel in world coordinates
        center_in_raster_px = np.ascontiguousarray(np.asarray(self.raster_size) * (0.5, 0.5))
        center_in_world_m = transform_point_nb(center_in_raster_px, world_from_raster)

        sem_im = self.render_semantic_map(center_in_world_m, raster_from_world, history_tl_faces[0])
        return sem_im.astype(np.float32) / 255

    def render_semantic_map(
        self, center_in_world: np.ndarray, raster_from_world: np.ndarray, tl_faces: np.ndarray
    ) -> np.ndarray:
        """Renders the semantic map at given x,y coordinates.

        Args:
            center_in_world (np.ndarray): XY of the image center in world ref system
            raster_from_world (np.ndarray):
        Returns:
            np.ndarray: RGB raster

        """

        # with timer_ms("np1"):
        img = 255 * np.ones(shape=(self.raster_size[1], self.raster_size[0], 3), dtype=np.uint8)

        # filter using half a radius from the center
        raster_radius = float(np.linalg.norm(self.raster_size * self.pixel_size)) / 2

        # get active traffic light faces
        active_tl_ids = set(filter_tl_faces_by_status(tl_faces, "ACTIVE")["face_id"].tolist())

        # plot lanes
        lanes_lines = defaultdict(list)

        lanes_area_list = []
        indices = elements_within_bounds_nb(center_in_world, self.bounds_info["lanes"]["bounds"], raster_radius)
        # with timer_ms("np2"):
        lanes_area_list = cv2_subpixel_transform_nb_indices(
            indices, self.xyz_array, raster_from_world, CV2_SHIFT_VALUE
        )

        # xy_left_list = cv2_subpixel_transform_nb_indices(
        #     indices, self.xyz_left_array, raster_from_world, CV2_SHIFT_VALUE
        # )
        # xy_right_list = cv2_subpixel_transform_nb_indices(
        #     indices, self.xyz_right_array, raster_from_world, CV2_SHIFT_VALUE
        # )
        # with timer_ms("np3"):
        for i, idx in enumerate(indices):
            # Note(lberg): this called on all polygons skips some of them, don't know why
            lane_type = "default"  # no traffic light face is controlling this lane
            # lane_tl_ids = set([MapAPI.id_as_str(la_tc) for la_tc in lane.traffic_controls])
            lane_tl_ids = self.idx_to_traffic_control_id_list[idx]
            for tl_id in lane_tl_ids.intersection(active_tl_ids):
                if self.proto_API.is_traffic_face_colour(tl_id, "red"):
                    lane_type = "red"
                elif self.proto_API.is_traffic_face_colour(tl_id, "green"):
                    lane_type = "green"
                elif self.proto_API.is_traffic_face_colour(tl_id, "yellow"):
                    lane_type = "yellow"

            lanes_area = lanes_area_list[i]
            # N = lanes_area.shape[0] // 2
            N = self.xyz_left_length[idx]
            lanes_lines[lane_type].extend([lanes_area[:N], lanes_area[N:]])

        # with timer_ms("cv"):
        # This does not work
        # cv2.fillPoly(img, lanes_area_list, (17, 17, 31), lineType=cv2.LINE_AA, shift=CV2_SHIFT)
        for lanes_area in lanes_area_list:
            cv2.fillPoly(img, [lanes_area], (17, 17, 31), lineType=cv2.LINE_AA, shift=CV2_SHIFT)
            # cv2.fillConvexPoly(img, lanes_area, (17, 17, 31), lineType=cv2.LINE_AA, shift=CV2_SHIFT)
            # print("lanes_area", lanes_area.shape)

        # For debugging
        # import IPython; IPython.embed()
        cv2.polylines(img, lanes_lines["default"], False, (255, 217, 82), lineType=cv2.LINE_AA, shift=CV2_SHIFT)
        cv2.polylines(img, lanes_lines["green"], False, (0, 255, 0), lineType=cv2.LINE_AA, shift=CV2_SHIFT)
        cv2.polylines(img, lanes_lines["yellow"], False, (255, 255, 0), lineType=cv2.LINE_AA, shift=CV2_SHIFT)
        cv2.polylines(img, lanes_lines["red"], False, (255, 0, 0), lineType=cv2.LINE_AA, shift=CV2_SHIFT)

        # plot crosswalks
        indices = elements_within_bounds_nb(center_in_world, self.bounds_info["crosswalks"]["bounds"], raster_radius)
        crosswalks = list(cv2_subpixel_transform_nb_indices(
            indices, self.crosswalk_2d_array, raster_from_world, CV2_SHIFT_VALUE
        ))
        cv2.polylines(img, crosswalks, True, (255, 117, 69), lineType=cv2.LINE_AA, shift=CV2_SHIFT)

        return img

    def to_rgb(self, in_im: np.ndarray, **kwargs: dict) -> np.ndarray:
        return (in_im * 255).astype(np.uint8)
