"""
Pangu 模型的 patch_size 扩展版本。

继承 onescience.models.pangu.Pangu，修复原版 __init__ 中
patchrecovery2d / patchrecovery3d 的 patch_size 硬编码问题，
使其从构造参数动态传入，支持 [2,6,6] 等非默认 patch_size。

用法：
    from pangu_patchsize import PanguPatchSize as Pangu
    替代原来的 from onescience.models.pangu import Pangu
"""

import math
import torch
import numpy as np
from torch import nn

from onescience.modules import (
    OneEmbedding,
    OneFuser,
    OneRecovery,
    OneSample,
)


class PanguPatchSize(nn.Module):
    """
    与 Pangu 完全等价，唯一区别：
    1. patchrecovery2d / patchrecovery3d 的 patch_size 从构造参数推导，
       而非硬编码为 (4,4) / (2,4,4)。
    2. 支持 layer2_depth / layer3_depth 参数进行 block 级剪枝。
       剪枝时保留前 N 个 block（block0 ~ block(N-1)），
       后续 block 被移除，减少计算量与参数量。
    """

    def __init__(
        self,
        img_size=(721, 1440),
        patch_size=(2, 6, 6),
        embed_dim=192,
        num_heads=(6, 12, 12, 6),
        window_size=(2, 6, 12),
        layer2_depth=6,
        layer3_depth=6,
    ):
        super().__init__()
        drop_path = np.linspace(0, 0.2, 8).tolist()
        self.layer2_depth = layer2_depth
        self.layer3_depth = layer3_depth

        # ---- Patch Embedding（与原版一致）----
        self.patchembed2d = OneEmbedding(
            style="PanguEmbedding",
            img_size=img_size,
            patch_size=patch_size[1:],
            Variables=7,
            embed_dim=embed_dim,
        )
        self.patchembed3d = OneEmbedding(
            style="PanguEmbedding",
            img_size=(13, *img_size),
            patch_size=patch_size,
            Variables=5,
            embed_dim=embed_dim,
        )

        patched_input_shape = (
            8,
            math.ceil(img_size[0] / patch_size[1]),
            math.ceil(img_size[1] / patch_size[2]),
        )

        # ---- Transformer 主干（与原版一致）----
        self.layer1 = OneFuser(
            style="PanguFuser",
            dim=embed_dim,
            input_resolution=patched_input_shape,
            depth=2,
            num_heads=num_heads[0],
            window_size=window_size,
            drop_path=drop_path[:2],
        )

        patched_downsampled_shape = (
            8,
            math.ceil(patched_input_shape[1] / 2),
            math.ceil(patched_input_shape[2] / 2),
        )

        self.downsample = OneSample(
            style="PanguDownSample",
            in_dim=embed_dim,
            input_resolution=patched_input_shape,
            output_resolution=patched_downsampled_shape,
        )
        self.layer2 = OneFuser(
            style="PanguFuser",
            dim=embed_dim * 2,
            input_resolution=patched_downsampled_shape,
            depth=layer2_depth,
            num_heads=num_heads[1],
            window_size=window_size,
            drop_path=drop_path[2 : 2 + layer2_depth],
        )
        self.layer3 = OneFuser(
            style="PanguFuser",
            dim=embed_dim * 2,
            input_resolution=patched_downsampled_shape,
            depth=layer3_depth,
            num_heads=num_heads[2],
            window_size=window_size,
            drop_path=drop_path[2 : 2 + layer3_depth],
        )
        self.upsample = OneSample(
            style="PanguUpSample",
            in_dim=embed_dim * 2,
            out_dim=embed_dim,
            input_resolution=patched_downsampled_shape,
            output_resolution=patched_input_shape,
        )
        self.layer4 = OneFuser(
            style="PanguFuser",
            dim=embed_dim,
            input_resolution=patched_input_shape,
            depth=2,
            num_heads=num_heads[3],
            window_size=window_size,
            drop_path=drop_path[:2],
        )

        # ---- Patch Recovery（修复硬编码：从 patch_size 参数推导）----
        self.patchrecovery2d = OneRecovery(
            style="PanguPatchRecovery",
            img_size=(721, 1440),
            patch_size=patch_size[1:],          # 原版硬编码为 (4,4)，改为动态
            in_chans=embed_dim * 2,
            out_chans=4,
        )
        self.patchrecovery3d = OneRecovery(
            style="PanguPatchRecovery",
            img_size=(13, 721, 1440),
            patch_size=patch_size,              # 原版硬编码为 (2,4,4)，改为动态
            in_chans=embed_dim * 2,
            out_chans=5,
        )

    def forward(self, x):
        """
        与原版 Pangu.forward 完全一致。
        """
        SurfaceInput = x[:, :7, :, :]
        UpperAirInput = x[:, 7:, :, :].reshape(x.shape[0], 5, 13, x.shape[2], x.shape[3])

        SurfaceFeatures = self.patchembed2d(SurfaceInput)
        UpperAirFeatures = self.patchembed3d(UpperAirInput)

        CombinedFeatures = torch.concat(
            [SurfaceFeatures.unsqueeze(2), UpperAirFeatures], dim=2
        )
        Batch, Channels, PressureLevels, Height, Width = CombinedFeatures.shape
        Tokens = CombinedFeatures.reshape(Batch, Channels, -1).transpose(1, 2)

        Tokens = self.layer1(Tokens)
        SkipTokens = Tokens

        Tokens = self.downsample(Tokens)
        Tokens = self.layer2(Tokens)
        Tokens = self.layer3(Tokens)
        Tokens = self.upsample(Tokens)
        Tokens = self.layer4(Tokens)

        OutputFeatures = torch.concat([Tokens, SkipTokens], dim=-1)
        OutputFeatures = OutputFeatures.transpose(1, 2).reshape(
            Batch, -1, PressureLevels, Height, Width
        )
        output_surface = OutputFeatures[:, :, 0, :, :]
        output_upper_air = OutputFeatures[:, :, 1:, :, :]

        output_surface = self.patchrecovery2d(output_surface)
        output_upper_air = self.patchrecovery3d(output_upper_air)
        return output_surface, output_upper_air
