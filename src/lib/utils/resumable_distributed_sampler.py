import torch
from torch.utils.data.distributed import DistributedSampler


class ResumableDistributedSampler(DistributedSampler):

    def __iter__(self):
        if not hasattr(self, "seed"):
            # For pytorch==1.5.0
            self.seed = 0
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        if hasattr(self, "iteration"):
            indices = indices[self.iteration:]

        return iter(indices)

    def __len__(self) -> int:
        if hasattr(self, "iteration"):
            return self.num_samples - self.iteration
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self, "iteration"):
            delattr(self, "iteration")
        self.epoch = epoch

    def resume(self, iteration: int, epoch: int) -> None:
        self.iteration = iteration
        self.epoch = epoch
