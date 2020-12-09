from functools import lru_cache
from typing import Iterator, Sequence, Union, no_type_check, Tuple

import numba as nb
import numpy as np
from numpy import radians, cos, sin
import pymap3d as pm
from l5kit.data import MapAPI

from l5kit.geometry import transform_points
from l5kit.data.proto.road_network_pb2 import GeoFrame, GlobalId, MapElement, MapFragment

from lib.utils.numba_utils import transform_points_nb

CACHE_SIZE = int(1e5)
ENCODING = "utf-8"


@nb.jit(nb.types.Tuple((nb.float64, nb.float64, nb.float64))(
    nb.float64, nb.float64, nb.int64
), nopython=True, nogil=True)
def geodetic2ecef(lat, lon, alt):
    # Assumes ell model = "wgs84"
    semimajor_axis = 6378137.0
    semiminor_axis = 6356752.31424518

    # radius of curvature of the prime vertical section
    N = semimajor_axis ** 2 / np.sqrt(semimajor_axis ** 2 * cos(lat) ** 2 + semiminor_axis ** 2 * sin(lat) ** 2)
    # Compute cartesian (geocentric) coordinates given  (curvilinear) geodetic
    # coordinates.
    x = (N + alt) * cos(lat) * cos(lon)
    y = (N + alt) * cos(lat) * sin(lon)
    z = (N * (semiminor_axis / semimajor_axis) ** 2 + alt) * sin(lat)
    return x, y, z


@nb.jit(nb.types.Tuple((nb.float64[:], nb.float64[:], nb.float64[:]))(
    nb.float64[:], nb.float64[:], nb.float64[:], nb.float64, nb.float64,
), nopython=True, nogil=True)
def enu2uvw(
    east: np.ndarray, north: np.ndarray, up: np.ndarray, lat0: np.ndarray, lon0: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = cos(lat0) * up - sin(lat0) * north
    w = sin(lat0) * up + cos(lat0) * north

    u = cos(lon0) * t - sin(lon0) * east
    v = sin(lon0) * t + cos(lon0) * east
    return u, v, w


@nb.jit(nb.types.Tuple((nb.float64[:], nb.float64[:], nb.float64[:]))(
    nb.float64[:], nb.float64[:], nb.float64[:], nb.float64, nb.float64, nb.int64
), nopython=True, nogil=True)
def enu2ecef(e1, n1, u1, lat0, lon0, h0):
    # Assuming `ell = None, deg = True`
    lat0 = radians(lat0)
    lon0 = radians(lon0)

    x0, y0, z0 = geodetic2ecef(lat0, lon0, h0)
    dx, dy, dz = enu2uvw(e1, n1, u1, lat0, lon0)

    return x0 + dx, y0 + dy, z0 + dz


@nb.jit(nb.float64[:, :](
    nb.int64[:], nb.int64[:], nb.int64[:], nb.int64, nb.int64, nb.float64[:, :]
), nopython=True, nogil=True)
def _unpack_deltas_cm_nb(dx, dy, dz, lat, lng, ecef_to_world):
    x = np.cumsum(dx / 100)
    y = np.cumsum(dy / 100)
    z = np.cumsum(dz / 100)
    frame_lat, frame_lng = lat / 1e7, lng / 1e7
    # xyz = np.stack(pm.enu2ecef(x, y, z, frame_lat, frame_lng, 0), axis=-1)
    xyz = np.stack(enu2ecef(x, y, z, frame_lat, frame_lng, 0), axis=-1)
    xyz = transform_points_nb(xyz, ecef_to_world)
    return xyz


class TunedMapAPI(MapAPI):
    def __init__(self, protobuf_map_path: str, world_to_ecef: np.ndarray):
        """
        Interface to the raw protobuf map file with the following features:
        - access to element using ID is O(1);
        - access to coordinates in world ref system for a set of elements is O(1) after first access (lru cache)
        - object support iteration using __getitem__ protocol

        Args:
            protobuf_map_path (str): path to the protobuf file
            world_to_ecef (np.ndarray): transformation matrix from world coordinates to ECEF (dataset dependent)
        """
        super(TunedMapAPI, self).__init__(protobuf_map_path, world_to_ecef)
        # self.protobuf_map_path = protobuf_map_path
        # self.ecef_to_world = np.linalg.inv(world_to_ecef)
        #
        # with open(protobuf_map_path, "rb") as infile:
        #     mf = MapFragment()
        #     mf.ParseFromString(infile.read())
        #
        # self.elements = mf.elements
        # self.ids_to_el = {self.id_as_str(el.id): idx for idx, el in enumerate(self.elements)}  # store a look-up table

    @no_type_check
    def unpack_deltas_cm(self, dx: Sequence[int], dy: Sequence[int], dz: Sequence[int], frame: GeoFrame) -> np.ndarray:
        """
        Get coords in world reference system (local ENU->ECEF->world).
        See the protobuf annotations for additional information about how coordinates are stored

        Args:
            dx (Sequence[int]): X displacement in centimeters in local ENU
            dy (Sequence[int]): Y displacement in centimeters in local ENU
            dz (Sequence[int]): Z displacement in centimeters in local ENU
            frame (GeoFrame): geo-location information for the local ENU. It contains lat and long origin of the frame

        Returns:
            np.ndarray: array of shape (Nx3) with XYZ coordinates in world ref system

        """
        xyz = _unpack_deltas_cm_nb(
            np.asarray(dx), np.asarray(dy), np.asarray(dz),
            frame.origin.lat_e7, frame.origin.lng_e7, self.ecef_to_world)
        return xyz

    @lru_cache(maxsize=CACHE_SIZE)
    def get_lane_coords(self, element_id: str) -> dict:
        """
        Get XYZ coordinates in world ref system for a lane given its id
        lru_cached for O(1) access

        Args:
            element_id (str): lane element id

        Returns:
            dict: a dict with the two boundaries coordinates as (Nx3) XYZ arrays
        """
        element = self[element_id]
        assert self.is_lane(element)

        lane = element.element.lane
        left_boundary = lane.left_boundary
        right_boundary = lane.right_boundary

        xyz_left = self.unpack_deltas_cm(
            left_boundary.vertex_deltas_x_cm,
            left_boundary.vertex_deltas_y_cm,
            left_boundary.vertex_deltas_z_cm,
            lane.geo_frame,
        )
        xyz_right = self.unpack_deltas_cm(
            right_boundary.vertex_deltas_x_cm,
            right_boundary.vertex_deltas_y_cm,
            right_boundary.vertex_deltas_z_cm,
            lane.geo_frame,
        )

        return {"xyz_left": xyz_left, "xyz_right": xyz_right}

    @lru_cache(maxsize=CACHE_SIZE)
    def get_crosswalk_coords(self, element_id: str) -> dict:
        """
        Get XYZ coordinates in world ref system for a crosswalk given its id
        lru_cached for O(1) access

        Args:
            element_id (str): crosswalk element id

        Returns:
            dict: a dict with the polygon coordinates as an (Nx3) XYZ array
        """
        element = self[element_id]
        assert self.is_crosswalk(element)
        traffic_element = element.element.traffic_control_element

        xyz = self.unpack_deltas_cm(
            traffic_element.points_x_deltas_cm,
            traffic_element.points_y_deltas_cm,
            traffic_element.points_z_deltas_cm,
            traffic_element.geo_frame,
        )

        return {"xyz": xyz}
