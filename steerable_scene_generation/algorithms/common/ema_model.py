import copy

import torch
import torch.nn as nn

from torch.nn.modules.batchnorm import _BatchNorm


class EMABaseModel(torch.nn.Module):
    """
    Exponential Moving Average of models weights.
    """

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def step(self, new_model):
        raise NotImplementedError()

    @property
    def model(self):
        raise NotImplementedError()


class EMAModel(EMABaseModel):
    """
    This has been adapted from
    https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/ema_model.py
    """

    def __init__(
        self,
        model: nn.Module,
        update_after_step: int = 0,
        inv_gamma: float = 1.0,
        power: float = 2.0 / 3.0,
        min_value: float = 0.0,
        max_value: float = 0.9999,
    ):
        """
        @crowsonkb's notes on EMA Warmup:
            If gamma=1 and power=1, implements a simple average. gamma=1,
            power=2/3 are good values for models you plan to train for a
            million or more steps (reaches decay factor 0.999 at 31.6K steps,
            0.9999 at 1M steps), gamma=1, power=3/4 for models you plan to
            train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
            215.4k steps).

        Args:
            inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
            power (float): Exponential factor of EMA warmup. Default: 2/3.
            min_value (float): The minimum EMA decay rate. Default: 0.
        """

        super().__init__()
        self.averaged_model = copy.deepcopy(model)
        self.averaged_model.eval()
        self.averaged_model.requires_grad_(False)

        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value

        self.register_buffer("optimization_step", torch.zeros(1, dtype=torch.long))

    @property
    def model(self) -> nn.Module:
        return self.averaged_model

    def get_decay(self, optimization_step: int) -> float:
        """
        Compute the decay factor for the exponential moving average.
        """
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power

        if step <= 0:
            return 0.0

        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model: nn.Module) -> None:
        decay = self.get_decay(self.optimization_step.item())

        for module, ema_module in zip(
            new_model.modules(), self.averaged_model.modules()
        ):
            if isinstance(module, _BatchNorm):
                assert isinstance(ema_module, _BatchNorm)
                ema_module.running_mean.mul_(decay)
                ema_module.running_mean.add_(module.running_mean, alpha=1 - decay)

                ema_module.running_var.mul_(decay)
                ema_module.running_var.add_(module.running_var, alpha=1 - decay)

            for param, ema_param in zip(
                module.parameters(recurse=False),
                ema_module.parameters(recurse=False),
            ):
                # Iterative over immediate parameters only.
                if isinstance(param, dict):
                    raise RuntimeError("Dict parameter not supported")

                if not param.requires_grad:
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                else:
                    ema_param.mul_(decay)
                    ema_param.add_(
                        param.data.to(dtype=ema_param.dtype), alpha=1 - decay
                    )

        self.optimization_step += 1
