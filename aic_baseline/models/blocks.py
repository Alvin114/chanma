from __future__ import annotations

import torch
import torch.nn as nn


def autopad(kernel_size: int, padding: int | None = None) -> int:
    return kernel_size // 2 if padding is None else padding


def make_divisible(value: float, divisor: int = 8) -> int:
    return max(divisor, int(value + divisor / 2) // divisor * divisor)


class Conv(nn.Module):
    def __init__(self, c1: int, c2: int, kernel: int = 1, stride: int = 1, padding: int | None = None, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel, stride, autopad(kernel, padding), groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1: int, c2: int, shortcut: bool = True, expansion: float = 0.5):
        super().__init__()
        hidden = int(c2 * expansion)
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.cv2 = Conv(hidden, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.cv2(self.cv1(x))
        return x + output if self.add else output


class C3(nn.Module):
    def __init__(self, c1: int, c2: int, repeats: int = 1, shortcut: bool = True, expansion: float = 0.5):
        super().__init__()
        hidden = int(c2 * expansion)
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.cv2 = Conv(c1, hidden, 1, 1)
        self.cv3 = Conv(2 * hidden, c2, 1, 1)
        self.m = nn.Sequential(*(Bottleneck(hidden, hidden, shortcut, expansion=1.0) for _ in range(repeats)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPF(nn.Module):
    def __init__(self, c1: int, c2: int, kernel: int = 5):
        super().__init__()
        hidden = c1 // 2
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.cv2 = Conv(hidden * 4, c2, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size=kernel, stride=1, padding=kernel // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        return self.cv2(torch.cat((x, y1, y2, self.pool(y2)), dim=1))


class CSPBackbone(nn.Module):
    """YOLOv5 CSPDarknet backbone returning P3, P4 and P5 features."""

    def __init__(self, in_channels: int, width: float = 0.5, depth: float = 0.33):
        super().__init__()
        channels = [make_divisible(value * width) for value in (64, 128, 256, 512, 1024)]
        repeats = [max(round(value * depth), 1) for value in (3, 6, 9, 3)]
        self.out_channels = (channels[2], channels[3], channels[4])
        self.layers = nn.ModuleList(
            [
                Conv(in_channels, channels[0], 6, 2, 2),
                Conv(channels[0], channels[1], 3, 2),
                C3(channels[1], channels[1], repeats[0]),
                Conv(channels[1], channels[2], 3, 2),
                C3(channels[2], channels[2], repeats[1]),
                Conv(channels[2], channels[3], 3, 2),
                C3(channels[3], channels[3], repeats[2]),
                Conv(channels[3], channels[4], 3, 2),
                C3(channels[4], channels[4], repeats[3]),
                SPPF(channels[4], channels[4], 5),
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p3 = p4 = p5 = None
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index == 4:
                p3 = x
            elif index == 6:
                p4 = x
            elif index == 9:
                p5 = x
        return p3, p4, p5


class PANNeck(nn.Module):
    def __init__(self, channels: tuple[int, int, int], depth: float = 0.33):
        super().__init__()
        c3, c4, c5 = channels
        repeats = max(round(3 * depth), 1)
        self.reduce_p5 = Conv(c5, c4, 1, 1)
        self.c3_p4 = C3(c4 + c4, c4, repeats, shortcut=False)
        self.reduce_p4 = Conv(c4, c3, 1, 1)
        self.c3_p3 = C3(c3 + c3, c3, repeats, shortcut=False)
        self.down_p3 = Conv(c3, c3, 3, 2)
        self.c3_n4 = C3(c3 + c3, c4, repeats, shortcut=False)
        self.down_p4 = Conv(c4, c4, 3, 2)
        self.c3_n5 = C3(c4 + c4, c5, repeats, shortcut=False)

    def forward(self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        p3, p4, p5 = features
        p5_reduced = self.reduce_p5(p5)
        p4_fused = self.c3_p4(torch.cat((nn.functional.interpolate(p5_reduced, scale_factor=2, mode="nearest"), p4), 1))
        p4_reduced = self.reduce_p4(p4_fused)
        out3 = self.c3_p3(torch.cat((nn.functional.interpolate(p4_reduced, scale_factor=2, mode="nearest"), p3), 1))
        out4 = self.c3_n4(torch.cat((self.down_p3(out3), p4_reduced), 1))
        out5 = self.c3_n5(torch.cat((self.down_p4(out4), p5_reduced), 1))
        return out3, out4, out5

