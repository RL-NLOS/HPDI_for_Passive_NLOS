"""Shared U-shaped implementation for the RN and IRN branches of HPDI."""

import torch
from torch import nn


class DS_ResidualBlock(nn.Module):
    def __init__(self, out_channels) -> None:
        super().__init__()
        self.in_ch = out_channels
        self.out_ch = out_channels
        self.net = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Identity()
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.net(x)
        y = self.lrelu(y + self.shortcut(x))
        return y

    def flops(self, h, w):
        return (
            h * w * self.out_ch * 3 * 3 * self.in_ch
            + h * w * self.out_ch * 1 * 1 * self.in_ch
            + h * w * self.out_ch * 3 * 3 * self.out_ch
        )


class US_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.net(x)
        y = self.lrelu(y + self.shortcut(x))
        return y

    def flops(self, h, w):
        return (
            h * w * self.out_ch * 3 * 3 * self.in_ch
            + h * w * self.out_ch * 1 * 1 * self.in_ch
            + h * w * self.out_ch * 3 * 3 * self.out_ch
        )


class EncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.pool_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.feat_layer = DS_ResidualBlock(out_channels)

    def forward(self, x):
        return self.feat_layer(self.pool_layer(x))

    def flops(self, h, w):
        return (
            self.feat_layer.flops(h / 2, w / 2) + h / 2 * w / 2 * self.out_ch * self.in_ch * 4 * 4
        )


class DecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, up_token="upconv") -> None:
        super().__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels
        if up_token == "deconv":
            self.up_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            self.up_layer = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            )
        self.feat_layer = US_ResidualBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up_layer(x1)
        xc = torch.cat([x1, x2], dim=1)
        return self.feat_layer(xc)

    def flops(self, H, W):
        return 2 * H * 2 * W * self.out_ch * 3 * 3 * self.in_ch + self.feat_layer.flops(
            2 * H, 2 * W
        )


class DecoderLayer_final(nn.Module):
    def __init__(self, in_channels, out_channels, up_token="upconv") -> None:
        super().__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels
        if up_token == "deconv":
            self.up_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            self.up_layer = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            )
        self.feat_layer = US_ResidualBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.up_layer(x)
        return self.feat_layer(x)

    def flops(self, H, W):
        return 2 * H * 2 * W * self.out_ch * 3 * 3 * self.in_ch + self.feat_layer.flops(
            2 * H, 2 * W
        )


class _ReconstructionBackbone(nn.Module):
    """Residual U-Net returning a reconstruction and multiscale decoder features."""

    def __init__(
        self,
        in_ch,
        out_ch,
        img_size=32,
        base_channels=32,
        half_block=3,
        encode_stage=EncoderLayer,
        decode_stage=DecoderLayer,
    ):
        super().__init__()
        self.img_size = img_size
        self.base_ch = base_channels
        self.half_block = half_block
        self.down_channels_list = [base_channels * 2**num for num in range(half_block + 1)]
        self.up_channels_list = list(reversed(self.down_channels_list))
        self.down_img_size_list = [img_size // 2**num for num in range(half_block + 1)]
        self.fir_layer = nn.Sequential(
            nn.Conv2d(in_ch, base_channels, kernel_size=3, padding=1), nn.LeakyReLU(inplace=True)
        )
        self.lat_layer = nn.Conv2d(base_channels, out_ch, kernel_size=3, padding=1)
        self.down_module_list = nn.ModuleList([])
        self.up_module_list = nn.ModuleList([])
        for idx in range(self.half_block):
            self.down_module_list.append(
                encode_stage(self.down_channels_list[idx], self.down_channels_list[idx + 1])
            )
            self.up_module_list.append(
                decode_stage(self.up_channels_list[idx], self.up_channels_list[idx + 1])
            )
        self.bottleneck_layer = DS_ResidualBlock(self.down_channels_list[-1])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fir_layer(x)
        encoder_out_list = [x]
        for encoder_layer in self.down_module_list:
            x = encoder_layer(x)
            encoder_out_list.append(x)
        encoder_out_list = list(reversed(encoder_out_list))
        x = self.bottleneck_layer(x)
        decoder_out_list = [x]
        for idx, decoder_layer in enumerate(self.up_module_list):
            x = decoder_layer(x, encoder_out_list[idx + 1])
            decoder_out_list.append(x)
        out = self.lat_layer(x)
        out = self.sigmoid(out)
        return (out, decoder_out_list)

    def flops(self):
        flops = 0
        flops += self.img_size * self.img_size * self.base_ch * 3 * 3 * 3
        flops += self.img_size * self.img_size * 3 * 3 * 3 * self.base_ch
        for enc, h in zip(self.down_module_list, self.down_img_size_list):
            flops += enc.flops(h, h)
        img_rev_list = list(reversed(self.down_img_size_list))
        for dec, h in zip(self.up_module_list, img_rev_list):
            flops += dec.flops(h, h)
        flops += self.bottleneck_layer.flops(img_rev_list[0], img_rev_list[0])
        return flops


class RefinementNet(_ReconstructionBackbone):
    """Refinement network (RN): refine LTPM reconstructions in the CTFP."""


class ImplicitReconstructionNet(_ReconstructionBackbone):
    """Implicit reconstruction network (IRN): map RAW measurements to scenes."""
