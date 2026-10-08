"""Cross-attention fusion network; original forward operations are preserved."""

import torch
import torch.nn.functional as F
from torch import nn


class EfficientCrossAttention(nn.Module):
    def __init__(self, in_channels_x, in_channels_y, key_channels, head_count, value_channels):
        super().__init__()
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels
        self.keys = nn.Conv2d(in_channels_y, key_channels, 1)
        self.queries = nn.Conv2d(in_channels_x, key_channels, 1)
        self.values = nn.Conv2d(in_channels_y, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels_x, 1)

    def forward(self, x, y):
        n, _, h, w = x.size()
        keys = self.keys(y).reshape((n, self.key_channels, h * w))
        queries = self.queries(x).reshape(n, self.key_channels, h * w)
        values = self.values(y).reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count
        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=2)
            query = F.softmax(
                queries[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=1
            )
            value = values[:, i * head_value_channels : (i + 1) * head_value_channels, :]
            context = key @ value.transpose(1, 2)
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)
        aggregated_values = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated_values)
        return attention


class LayerNorm(nn.Module):
    """LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-06, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class FFN(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU(), drop=0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features,
        )
        self.act = act_layer
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CrossAttentionFusion(nn.Module):
    """Fuse IRN and RN features through bidirectional cross-attention and an FFN."""

    def __init__(self, dim=(32, 32)):
        super().__init__()
        self.dim_ev, self.dim_img = dim
        self.norm_ev = LayerNorm(normalized_shape=self.dim_img, data_format="channels_first")
        self.norm_img = LayerNorm(normalized_shape=self.dim_img, data_format="channels_first")
        self.i2e = EfficientCrossAttention(
            in_channels_x=self.dim_img,
            in_channels_y=self.dim_img,
            key_channels=self.dim_img,
            head_count=4,
            value_channels=self.dim_img,
        )
        self.e2i = EfficientCrossAttention(
            in_channels_x=self.dim_img,
            in_channels_y=self.dim_img,
            key_channels=self.dim_img,
            head_count=4,
            value_channels=self.dim_img,
        )
        self.norm_ffn = LayerNorm(normalized_shape=self.dim_img, data_format="channels_first")
        self.ffn = FFN(in_features=self.dim_img, hidden_features=self.dim_img // 4)

    def forward(self, ev, img):
        ev_f, img_f = (self.norm_ev(ev), self.norm_img(img))
        ev_f = self.i2e(ev_f, img_f) + ev
        img_f = self.e2i(img_f, ev_f) + img
        out = ev_f + img_f
        out = self.ffn(self.norm_ffn(out)) + out
        return out


class DecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, cat=True, up_token="upconv") -> None:
        super().__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.cat = cat
        if up_token == "deconv":
            self.up_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            self.up_layer = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            )

    def forward(self, x1, x2):
        if self.cat:
            xc = torch.cat([x1, x2], dim=1)
        else:
            xc = x1
        return self.up_layer(xc)


class AdaptiveFeatureSelection(nn.Module):
    """Select features from both branches with residual sigmoid channel gates."""

    def __init__(self, in_channels, M=2, r=4):
        """Gate two feature branches with channel attention. M defaults to 2."""
        super().__init__()
        d = in_channels * 2 // r
        self.M = M
        self.adaptive_avg_pool2d = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels * 2, d, 1, bias=False), nn.LeakyReLU(0.2, inplace=True)
        )
        self.fcs = nn.ModuleList()
        for i in range(M):
            self.fcs.append(nn.Conv2d(d, in_channels, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        U = torch.cat([x, y], dim=1)
        s = self.adaptive_avg_pool2d(U)
        z = self.fc(s)
        weights = [fc(z) for fc in self.fcs]
        return (self.sigmoid(weights[0]) * x + x, self.sigmoid(weights[1]) * y + y)


class FusionNet(nn.Module):
    """Fusion network (FN): fuse the four decoder feature scales from IRN and RN."""

    def __init__(self, out_ch, img_size=32, base_channels=32, half_block=3):
        super().__init__()
        self.img_size = img_size
        self.base_ch = base_channels
        self.half_block = half_block
        self.down_channels_list = [base_channels * 2**num for num in range(half_block + 1)]
        self.up_channels_list = list(reversed(self.down_channels_list))
        self.down_img_size_list = [img_size // 2**num for num in range(half_block + 1)]
        self.att0 = AdaptiveFeatureSelection(256)
        self.att1 = AdaptiveFeatureSelection(128)
        self.att2 = AdaptiveFeatureSelection(64)
        self.att3 = AdaptiveFeatureSelection(32)
        self.fusion_module_1 = CrossAttentionFusion(dim=[base_channels * 8, base_channels * 8])
        self.fusion_module_2 = CrossAttentionFusion(dim=[base_channels * 4, base_channels * 4])
        self.fusion_module_3 = CrossAttentionFusion(dim=[base_channels * 2, base_channels * 2])
        self.fusion_module_4 = CrossAttentionFusion(dim=[base_channels * 1, base_channels * 1])
        self.up_conv_1 = DecoderLayer(base_channels * 8, base_channels * 4, cat=False)
        self.up_conv_2 = DecoderLayer(base_channels * 8, base_channels * 2, cat=True)
        self.up_conv_3 = DecoderLayer(base_channels * 4, base_channels, cat=True)
        self.lat_layer = nn.Conv2d(base_channels * 2, out_ch, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, f_x, f_y):
        self.att0_x, self.att0_y = self.att0(f_x[0], f_y[0])
        z1 = self.fusion_module_1(self.att0_x, self.att0_y)
        z1 = self.up_conv_1(z1, z1)
        self.att1_x, self.att1_y = self.att1(f_x[1], f_y[1])
        z2 = self.fusion_module_2(self.att1_x, self.att1_y)
        z2 = self.up_conv_2(z2, z1)
        self.att2_x, self.att2_y = self.att2(f_x[2], f_y[2])
        z3 = self.fusion_module_3(self.att2_x, self.att2_y)
        z3 = self.up_conv_3(z3, z2)
        self.att3_x, self.att3_y = self.att3(f_x[3], f_y[3])
        z4 = self.fusion_module_4(self.att3_x, self.att3_y)
        out = self.lat_layer(torch.cat([z4, z3], dim=1))
        out = self.sigmoid(out)
        return out

