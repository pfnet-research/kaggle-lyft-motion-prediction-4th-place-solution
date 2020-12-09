from logging import getLogger

from torch import nn


class EMA(object):
    """Exponential moving average of model parameters.

    Ref
     - https://github.com/tensorflow/addons/blob/v0.10.0/tensorflow_addons/optimizers/moving_average.py#L26-L103
     - https://anmoljoshi.com/Pytorch-Dicussions/

    Args:
        model (nn.Module): Model with parameters whose EMA will be kept.
        decay (float): Decay rate for exponential moving average.
        strict (bool): Apply strict check for `assign` & `resume`.
        use_dynamic_decay (bool): Dynamically change decay rate. If `True`, small decay rate is
            used at the beginning of training to move moving average faster.
    """  # NOQA

    def __init__(
        self,
        model: nn.Module,
        decay: float,
        strict: bool = True,
        use_dynamic_decay: bool = True,
    ):
        self.decay = decay
        self.model = model
        self.strict = strict
        self.use_dynamic_decay = use_dynamic_decay
        self.logger = getLogger(__name__)
        self.n_step = 0

        self.shadow = {}
        self.original = {}

        # Flag to manage which parameter is assigned.
        # When `False`, original model's parameter is used.
        # When `True` (`assign` method is called), `shadow` parameter (ema param) is used.
        self._assigned = False

        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def step(self):
        self.n_step += 1
        if self.use_dynamic_decay:
            _n_step = float(self.n_step)
            decay = min(self.decay, (1.0 + _n_step) / (10.0 + _n_step))
        else:
            decay = self.decay

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    # alias
    __call__ = step

    def assign(self):
        """Assign exponential moving average of parameter values to the respective parameters."""
        if self._assigned:
            if self.strict:
                raise ValueError("[ERROR] `assign` is called again before `resume`.")
            else:
                self.logger.warning(
                    "`assign` is called again before `resume`."
                    "shadow parameter is already assigned, skip."
                )
                return

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]
        self._assigned = True

    def resume(self):
        """Restore original parameters to a model.

        That is, put back the values that were in each parameter at the last call to `assign`.
        """
        if not self._assigned:
            if self.strict:
                raise ValueError("[ERROR] `resume` is called before `assign`.")
            else:
                self.logger.warning("`resume` is called before `assign`, skip.")
                return

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]
        self._assigned = False
