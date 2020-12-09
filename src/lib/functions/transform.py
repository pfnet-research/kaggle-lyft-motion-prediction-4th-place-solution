import torch
from torch import Tensor


def transform_points_batch(points: Tensor, transf_matrix: Tensor) -> Tensor:
    """
    Transform points using transformation matrix.
    Note this function assumes points.shape[1] == matrix.shape[1] - 1, which means that the last row on the matrix
    does not influence the final result.
    For 2D points only the first 2x3 part of the matrix will be used.

    Args:
        points (Tensor): Input points (Nx2).
        transf_matrix (Tensor): Nx3x3 transformation matrix for 2D input

    Returns:
        transformed_points (Tensor): array of shape (N,2) for 2D input points, or (N,3) points for 3D input points
    """
    bs, cdim = points.shape
    assert cdim == 2
    assert transf_matrix.shape == (bs, 3, 3)

    num_dims = 2
    transf_matrix = transf_matrix.transpose(1, 2)  # same with transf_matrix.permute(0, 2, 1)

    transf_points = torch.bmm(points.unsqueeze(1), transf_matrix[:, :num_dims, :num_dims]).squeeze(1)
    assert transf_points.shape == (bs, 2)
    return transf_points + transf_matrix[:, -1, :num_dims]
