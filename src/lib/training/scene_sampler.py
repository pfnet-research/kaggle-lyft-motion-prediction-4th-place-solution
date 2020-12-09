import math

import torch
from torch.utils.data import Sampler
import torch.distributed as dist
import numpy as np


def get_valid_starts_and_ends(get_frame_arguments: np.ndarray, min_state_index: int = 0):
    get_frame_arguments = get_frame_arguments[:]  # put on the memory if the array is zarr

    scene_change_points = np.where(np.diff(get_frame_arguments[:, 1], 1) > 0)[0] + 1
    starts = np.r_[0, scene_change_points]
    ends = np.r_[scene_change_points, len(get_frame_arguments)]

    valid_starts, valid_ends = [], []
    while len(starts) > 0:
        ok = get_frame_arguments[starts, 2] >= min_state_index
        valid_starts.append(starts[ok])
        valid_ends.append(ends[ok])
        starts, ends = starts[~ok], ends[~ok]

        starts += 1
        ok = starts < ends
        starts, ends = starts[ok], ends[ok]

    return np.concatenate(valid_starts), np.concatenate(valid_ends)


class SceneSampler(Sampler):

    def __init__(self, get_frame_arguments: np.ndarray, min_state_index: int = 0) -> None:
        self.starts, self.ends = get_valid_starts_and_ends(get_frame_arguments, min_state_index)

    def __len__(self) -> int:
        return len(self.starts)

    def __iter__(self):
        indices = np.random.permutation(len(self.starts))
        return iter(np.random.randint(self.starts[indices], self.ends[indices]))


class DistributedSceneSampler(Sampler):

    def __init__(
        self,
        get_frame_arguments: np.ndarray,
        min_state_index: int = 0,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.starts, self.ends = get_valid_starts_and_ends(get_frame_arguments, min_state_index)
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.starts) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.starts), generator=g).tolist()
        else:
            indices = list(range(len(self.starts)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(np.random.randint(self.starts[indices], self.ends[indices]))

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Arguments:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
