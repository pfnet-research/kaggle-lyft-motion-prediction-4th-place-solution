from typing import Mapping, Any

from torch import optim

from pytorch_pfn_extras.training.extension import Extension, PRIORITY_READER
from pytorch_pfn_extras.training.manager import ExtensionsManager


class LRScheduler(Extension):
    """A thin wrapper to resume the lr_scheduler"""

    trigger = 1, 'iteration'
    priority = PRIORITY_READER
    name = None

    def __init__(self, optimizer: optim.Optimizer, scheduler_type: str, scheduler_kwargs: Mapping[str, Any]) -> None:
        super().__init__()
        self.scheduler = getattr(optim.lr_scheduler, scheduler_type)(optimizer, **scheduler_kwargs)

    def __call__(self, manager: ExtensionsManager) -> None:
        self.scheduler.step()

    def state_dict(self) -> None:
        return self.scheduler.state_dict()

    def load_state_dict(self, to_load) -> None:
        self.scheduler.load_state_dict(to_load)
