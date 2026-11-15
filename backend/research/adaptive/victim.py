"""The victim: a small CNN trained on what the user would train on (reference + incoming), to measure whether an
attack works (attack success rate, ASR) as opposed to whether Veritas sees it."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(2)


def _net() -> nn.Module:
    def block(i, o):
        return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2)]
    return nn.Sequential(*block(3, 32), *block(32, 64), *block(64, 128), nn.Flatten(), nn.Linear(128 * 16, 10))


def _tensor(images: np.ndarray) -> torch.Tensor:
    return (torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255 - 0.5) / 0.25


def _augment(x: torch.Tensor) -> torch.Tensor:
    """Random crop (pad 4) and horizontal flip, per sample."""
    n, h, w = len(x), x.shape[2], x.shape[3]
    p = F.pad(x, (4, 4, 4, 4))
    dy, dx = np.random.randint(0, 9, (2, n))
    out = torch.stack([p[i, :, dy[i]:dy[i] + h, dx[i]:dx[i] + w] for i in range(n)])
    flip = torch.rand(n) < 0.5
    out[flip] = out[flip].flip(3)
    return out


def train(images: np.ndarray, y: np.ndarray, epochs: int, seed: int) -> nn.Module:
    torch.manual_seed(seed); np.random.seed(seed)
    net = _net()
    x, t = _tensor(images), torch.from_numpy(y)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    net.train()
    for _ in range(epochs):
        for b in torch.randperm(len(x)).split(128):
            opt.zero_grad()
            F.cross_entropy(net(_augment(x[b])), t[b]).backward()
            opt.step()
    return net.eval()


@torch.no_grad()
def predict(net: nn.Module, images: np.ndarray) -> np.ndarray:
    x = _tensor(images)
    return np.concatenate([net(b).argmax(1).numpy() for b in x.split(500)])


def evaluate(net, test_x, test_y, ctx, trigger, s, flip: bool) -> dict:
    """Clean accuracy and ASR: share of non-target (flips: source-class) test images predicted as target once triggered."""
    asr_set = test_x[test_y == ctx["source"]] if flip else test_x[test_y != ctx["target"]]
    hit = predict(net, trigger(asr_set, ctx, s)) == ctx["target"]
    return {"acc": float((predict(net, test_x) == test_y).mean()), "asr": float(hit.mean())}
