"""TerraMind encoder with a light segmentation decoder for the Polissia patches.

Sentinel-2 and ascending Sentinel-1 go through TerraMind's pretrained patch
embeddings (S2L2A band subset, S1GRD). Descending Sentinel-1 is a new modality
initialised from the S1GRD embedding; NISAR is a new, randomly initialised
modality. TerraMind has no L-band pretraining, so the NISAR pathway is learned
from the training patches only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# dataset channel index -> TerraMind band name / pretraining statistics (S2 in DN = reflectance * 10000)
S2_BANDS = {0: ("BLUE", 1503.317, 2141.107), 1: ("GREEN", 1718.197, 2038.973), 2: ("RED", 1853.91, 2134.138),
            3: ("NIR_BROAD", 3083.234, 1871.918), 4: ("SWIR_1", 2424.884, 1434.261), 5: ("SWIR_2", 1857.648, 1334.311)}
S1_STATS = {0: (-12.599, 5.195), 1: (-20.293, 5.890)}  # VV, VH in dB (S1GRD)
GROUPS = {"S2L2A": range(0, 6), "S1GRD": range(6, 8), "S1DESC": range(8, 10), "NISAR": range(10, 14)}
S1_KEY = "untok_sen1grd@224"


def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class TerraMindSeg(nn.Module):
    """Input: the dataset's z-scored tensor (B, C, H, W) with missing pixels at 0."""

    def __init__(self, channels: list[int], mean, std, classes: int, variant: str = "terramind_v1_small",
                 pretrained: bool = True, img_size: int = 256, layers=(2, 5, 8, 11), width: int = 256):
        super().__init__()
        from terratorch.registry import BACKBONE_REGISTRY

        self.layers = layers
        self.mods: list[tuple[str, list[int]]] = []  # modality name, positions in the input tensor
        for name, chans in GROUPS.items():
            pos = [channels.index(c) for c in chans if c in channels]
            if pos and len(pos) != len(chans):
                raise ValueError(f"TerraMind needs whole sensors; {name} is partial in {channels}")
            if pos:
                self.mods.append((name, pos))
        names = [n for n, _ in self.mods]
        if "S1DESC" in names and "S1GRD" not in names:  # a lone descending pass uses the pretrained S1 embedding
            self.mods = [("S1GRD" if n == "S1DESC" else n, p) for n, p in self.mods]
            names = [n for n, _ in self.mods]
        modalities = [n if n in ("S2L2A", "S1GRD") else {n: len(p)} for n, p in self.mods]
        bands = {"S2L2A": [S2_BANDS[i][0] for i in range(6)]} if "S2L2A" in names else None
        self.backbone = BACKBONE_REGISTRY.build(variant, pretrained=pretrained, img_size=img_size, modalities=modalities,
                                                bands=bands, merge_method="concat")
        emb = self.backbone.encoder_embeddings
        if pretrained and "S1DESC" in names:
            emb["S1DESC"].load_state_dict(emb[S1_KEY].state_dict())

        # convert dataset z-scores to TerraMind's pretraining normalisation: x * scale + shift
        mean, std = torch.as_tensor(mean, dtype=torch.float32), torch.as_tensor(std, dtype=torch.float32)
        scale, shift = torch.ones(len(channels)), torch.zeros(len(channels))
        for p, c in enumerate(channels):
            if c < 6:
                _, m, s = S2_BANDS[c]
                scale[p], shift[p] = std[p] * 1e4 / s, (mean[p] * 1e4 - m) / s
            elif c < 10:
                m, s = S1_STATS[(c - 6) % 2]
                scale[p], shift[p] = std[p] / s, (mean[p] - m) / s
        self.register_buffer("scale", scale.view(1, -1, 1, 1))
        self.register_buffer("shift", shift.view(1, -1, 1, 1))

        dim = self.backbone.out_channels[0]
        self.reduce = nn.ModuleList(nn.Sequential(nn.Conv2d(dim, width, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
                                    for _ in layers)
        self.fuse = conv_block(width * len(layers), width)
        self.up = nn.ModuleList(conv_block(a, b) for a, b in ((width, 128), (128, 96), (96, 64), (64, 48)))
        self.stem = nn.Sequential(conv_block(len(channels), 32), conv_block(32, 32))  # full-resolution detail
        self.head = nn.Sequential(conv_block(48 + 32, 48), nn.Conv2d(48, classes, 1))
        self.head_dropout = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        t = x * self.scale + self.shift
        feats = self.backbone({name: t[:, pos] for name, pos in self.mods})
        g = h // 16
        maps = [r(feats[i].transpose(1, 2).reshape(b, -1, g, w // 16)) for r, i in zip(self.reduce, self.layers)]
        y = self.fuse(torch.cat(maps, 1))
        for up in self.up:
            y = up(F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False))
        y = torch.cat([y, self.stem(x)], 1)
        return self.head(F.dropout2d(y, self.head_dropout, self.training))
