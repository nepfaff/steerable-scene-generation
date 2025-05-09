from typing import Tuple

import torch
import torch.nn as nn

from torch.utils.data import DataLoader


class AverageMeter:
    """Average computation class."""

    def __init__(self):
        self._value = 0
        self._cnt = 0

    def __iadd__(self, x):
        if torch.is_tensor(x):
            self._value += x.sum().item()
            self._cnt += x.numel()
        else:
            self._value += x
            self._cnt += 1
        return self

    @property
    def value(self):
        return self._value / self._cnt


@torch.no_grad()
def compute_loss_and_acc(
    dataloader: DataLoader, model: nn.Module, device: torch.device
) -> Tuple[float, float]:
    model = model.eval()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        y_hat = model(x)
        loss = torch.nn.functional.binary_cross_entropy(y_hat, y)
        acc = (torch.abs(y - y_hat) < 0.5).float().mean()
        loss_meter += loss
        acc_meter += acc

    return loss_meter.value, acc_meter.value
