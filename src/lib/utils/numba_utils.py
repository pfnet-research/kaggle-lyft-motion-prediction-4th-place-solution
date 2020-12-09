import numba as nb
import numpy as np


@nb.jit(nb.float64[:](nb.float64[:], nb.float64[:, :]), nopython=True, nogil=True)
def transform_point_nb(point: np.ndarray, transf_matrix: np.ndarray) -> np.ndarray:
    """ Transform a single vector using transformation matrix.

    Args:
        point (np.ndarray): vector of shape (N)
        transf_matrix (np.ndarray): transformation matrix of shape (N+1, N+1)

    Returns:
        np.ndarray: vector of same shape as input point
    """
    point_ext = np.ascontiguousarray(np.hstack((point, np.ones(1))))
    # (N+1, N+1) @ (N+1)
    # p = np.matmul(transf_matrix, point_ext)[: point.shape[0]]
    p = np.dot(np.ascontiguousarray(transf_matrix), point_ext)[: point.shape[0]]
    return p


@nb.jit(nb.float64[:, :](nb.float64[:, :], nb.float64[:, :]), nopython=True, nogil=True)
def transform_points_nb(points: np.ndarray, transf_matrix: np.ndarray) -> np.ndarray:
    """
    Transform points using transformation matrix.
    Note this function assumes points.shape[1] == matrix.shape[1] - 1, which means that the last row on the matrix
    does not influence the final result.
    For 2D points only the first 2x3 part of the matrix will be used.

    Args:
        points (np.ndarray): Input points (Nx2) or (Nx3).
        transf_matrix (np.ndarray): 3x3 or 4x4 transformation matrix for 2D and 3D input respectively

    Returns:
        np.ndarray: array of shape (N,2) for 2D input points, or (N,3) points for 3D input points
    """
    assert len(points.shape) == len(transf_matrix.shape) == 2
    assert transf_matrix.shape[0] == transf_matrix.shape[1]

    assert points.shape[1] in [2, 3]
    # if points.shape[1] not in [2, 3]:
    #     raise AssertionError("Points input should be (N, 2) or (N,3) shape, received {}".format(points.shape))
    # assert points.shape[1] == 2

    assert points.shape[1] == transf_matrix.shape[1] - 1, "points dim should be one less than matrix dim"

    num_dims = len(transf_matrix) - 1
    transf_matrix = transf_matrix.T

    # return points @ transf_matrix[:num_dims, :num_dims] + transf_matrix[-1, :num_dims]
    return np.dot(
        np.ascontiguousarray(points),
        np.ascontiguousarray(transf_matrix[:num_dims, :num_dims])
    ) + transf_matrix[-1, :num_dims]


# --- For SemanticRasterizer ---
@nb.jit(nb.int64[:](nb.float64[:], nb.float64[:, :, :], nb.float64), nopython=True, nogil=True)
def elements_within_bounds_nb(center: np.ndarray, bounds: np.ndarray, half_extent: float) -> np.ndarray:
    """
    Get indices of elements for which the bounding box described by bounds intersects the one defined around
    center (square with side 2*half_side)

    Args:
        center (float): XY of the center
        bounds (np.ndarray): array of shape Nx2x2 [[x_min,y_min],[x_max, y_max]]
        half_extent (float): half the side of the bounding box centered around center

    Returns:
        np.ndarray: indices of elements inside radius from center
    """
    x_center, y_center = center

    x_min_in = x_center > bounds[:, 0, 0] - half_extent
    y_min_in = y_center > bounds[:, 0, 1] - half_extent
    x_max_in = x_center < bounds[:, 1, 0] + half_extent
    y_max_in = y_center < bounds[:, 1, 1] + half_extent
    indices = np.nonzero(x_min_in & y_min_in & x_max_in & y_max_in)[0]
    return indices
