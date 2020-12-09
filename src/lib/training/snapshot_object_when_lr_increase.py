from typing import Any, Optional

from torch import optim
from pytorch_pfn_extras.training.extension import Extension, PRIORITY_READER
from pytorch_pfn_extras.training.manager import ExtensionsManager
from pytorch_pfn_extras.training.extensions import snapshot_object


class SnapshotObjectWhenLRIncrease(Extension):

    trigger = 1, 'iteration'
    priority = PRIORITY_READER
    name = None

    def __init__(
        self,
        target: Any,
        optimizer: optim.Optimizer,
        param_group: int = 0,
        filename: str = "snapshot_{cycle_count}th_cycle.pt",
        saver_rank: Optional[int] = None
    ) -> None:
        super().__init__()
        self.cycle_count = 0
        self.lr_before = float("inf")
        self.target = target
        self.optimizer = optimizer
        self.param_group = param_group
        self.filename = filename
        self.saver_rank = saver_rank

    def __call__(self, manager: ExtensionsManager) -> None:
        lr_after = self.optimizer.param_groups[self.param_group]['lr']
        if self.lr_before < lr_after:
            filename = self.filename.format(cycle_count=self.cycle_count)
            save_func = snapshot_object(self.target, filename, saver_rank=self.saver_rank)
            save_func(manager)
            self.cycle_count += 1
        self.lr_before = lr_after

    def state_dict(self) -> None:
        return {"cycle_count": self.cycle_count, "lr_before": self.lr_before}

    def load_state_dict(self, to_load) -> None:
        self.cycle_count = to_load["cycle_count"]
        self.lr_before = to_load["lr_before"]
