from typing import Union

import torch
from pytorch_pfn_extras.training.extensions import Evaluator


class DistributedEvaluator(Evaluator):
    def __init__(
        self,
        iterator,
        target,
        eval_hook=None,
        eval_func=None,
        local_rank=0,
        world_size=1,
        device: Union[torch.device, str] = "cpu",
        **kwargs,
    ):
        super(DistributedEvaluator, self).__init__(
            iterator, target, eval_hook=eval_hook, eval_func=eval_func, **kwargs
        )
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = device

    def evaluate(self):
        local_rank = self.local_rank
        world_size = self.world_size

        result = super(DistributedEvaluator, self).evaluate()
        print(f"[DEBUG] evaluate: local_rank {local_rank}, result {result}")
        keys_list = list(result.keys())
        keys_list.sort()  # To make order same for all process.
        for key in keys_list:
            value = torch.as_tensor(result[key], device=self.device)
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
            result[key] = float(value.item() / world_size)
        if local_rank == 0:
            print(
                f"[DEBUG] evaluate: After all_reduce: local_rank {local_rank}, result {result}"
            )
        return result
