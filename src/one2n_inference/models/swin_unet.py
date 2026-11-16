"""Three-dimensional Swin-UNet used for One2N inference.

Based on the author-provided Swin_Unet.py and the Swin Transformer design
(https://arxiv.org/abs/2103.14030). Parameter names and the manual attention
computation are retained for compatibility with the original checkpoint.
"""

import numpy as np
import torch
import torch.nn as nn
from timm.layers import DropPath, trunc_normal_


class Mlp(nn.Module):
    """Two-layer feed-forward branch."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, D, H, W, C = x.shape
    x = x.view(
        B,
        D // window_size[0],
        window_size[0],
        H // window_size[1],
        window_size[1],
        W // window_size[2],
        window_size[2],
        C,
    )
    windows = (
        x.permute(0, 1, 3, 5, 2, 4, 6, 7)
        .contiguous()
        .view(-1, window_size[0], window_size[1], window_size[2], C)
    )
    return windows


def window_reverse(windows, window_size, D, H, W):
    B = int(windows.shape[0] / (D * H * W / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(
        B,
        D // window_size[0],
        H // window_size[1],
        W // window_size[2],
        window_size[0],
        window_size[1],
        window_size[2],
        -1,
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window attention with three-dimensional relative position bias."""

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** (-0.5)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                num_heads,
            )
        )
        coords_d = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])
        coords = torch.stack(torch.meshgrid([coords_d, coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten.unsqueeze(2) - coords_flatten.unsqueeze(1)
        relative_coords += torch.from_numpy(window_size).unsqueeze(1).unsqueeze(2) - 1
        relative_position_index = (
            relative_coords[0] * (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
            + relative_coords[1] * (2 * window_size[2] - 1)
            + relative_coords[2]
        )
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (qkv[0], qkv[1], qkv[2])
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PatchEmbed(nn.Module):
    """Embed non-overlapping three-dimensional patches."""

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=384, norm_layer=None):
        super().__init__()
        patches_resolution = img_size // patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution.prod()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, D, H, W = x.shape
        assert (
            D == self.img_size[0] and H == self.img_size[1] and (W == self.img_size[2])
        ), f"Input image size ({D}*{H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchRecovery(nn.Module):
    """Recover fields using the original patch-offset channel order."""

    def __init__(self, img_size=224, patch_size=4, embed_dim=384, out_chans=192, norm_layer=None):
        super().__init__()
        patches_resolution = img_size // patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution.prod()
        self.mid_chans = out_chans * patch_size.prod()
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(embed_dim, self.mid_chans, kernel_size=1, stride=1)
        if norm_layer is not None:
            self.norm = norm_layer(self.mid_chans)
        else:
            self.norm = None

    def forward(self, x):
        B, _, C_ = x.shape
        D, H, W = self.img_size
        NPD, NPH, NPW = self.patches_resolution
        PSD, PSH, PSW = self.patch_size
        x = x.transpose(1, 2)
        x = x.view(B, C_, NPD, NPH, NPW)
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        x = x.transpose(1, 2)
        x = x.view(B, PSD, PSH, PSW, -1, NPD, NPH, NPW)
        x = x.permute(0, 4, 5, 1, 6, 2, 7, 3).contiguous()
        x = x.view(B, -1, D, H, W)
        return x


class SwinTransformerBlock(nn.Module):
    """Shifted-window attention with post-branch layer normalization."""

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        D, H, W = input_resolution
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert (self.shift_size <= self.window_size).all(), "shift_size must in 0-window_size"
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        if (self.shift_size > 0).any():
            D, H, W = self.input_resolution
            # Longitude remains periodic, matching the reference attention mask.
            img_mask = torch.zeros((1, D, H, W, 1))
            d_slices = (
                slice(0, -self.window_size[0]),
                slice(-self.window_size[0], -self.shift_size[0]),
                slice(-self.shift_size[0], None),
            )
            h_slices = (
                slice(0, -self.window_size[1]),
                slice(-self.window_size[1], -self.shift_size[1]),
                slice(-self.shift_size[1], None),
            )
            cnt = 0
            for d in d_slices:
                for h in h_slices:
                    img_mask[:, d, h, :, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(
                -1, self.window_size[0] * self.window_size[1] * self.window_size[2]
            )
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
                attn_mask == 0, float(0.0)
            )
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        D, H, W = self.input_resolution
        B, L, C = x.shape
        assert L == D * H * W, "input feature has wrong size"
        shortcut = x
        x = x.view(B, D, H, W, C)
        if (self.shift_size > 0).any():
            shifted_x = torch.roll(
                x,
                shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                dims=(1, 2, 3),
            )
            x_windows = window_partition(shifted_x, self.window_size)
        else:
            shifted_x = x
            x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(
            -1, self.window_size[0] * self.window_size[1] * self.window_size[2], C
        )
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(
            -1, self.window_size[0], self.window_size[1], self.window_size[2], C
        )
        if (self.shift_size > 0).any():
            shifted_x = window_reverse(attn_windows, self.window_size, D, H, W)
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                dims=(1, 2, 3),
            )
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, D, H, W)
            x = shifted_x
        x = x.view(B, D * H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        return x


class BasicLayer(nn.Module):
    """Transformer stage followed by optional patch embedding or recovery."""

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        updownsample=None,
        updown_size=1,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=window_size * 0 if i % 2 == 0 else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=(drop_path[i] if isinstance(drop_path, list) else drop_path),
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        if updownsample is PatchEmbed:
            self.updownsample = updownsample(
                img_size=input_resolution,
                patch_size=updown_size,
                in_chans=dim,
                embed_dim=2 * dim,
                norm_layer=norm_layer,
            )
        elif updownsample is PatchRecovery:
            self.updownsample = updownsample(
                img_size=input_resolution * updown_size,
                patch_size=updown_size,
                embed_dim=dim,
                out_chans=dim // 2,
                norm_layer=norm_layer,
            )
        else:
            self.updownsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.updownsample is not None:
            B, N, C = x.shape
            D, H, W = self.input_resolution
            if isinstance(self.updownsample, PatchEmbed):
                x = x.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
            x = self.updownsample(x)
            if isinstance(self.updownsample, PatchRecovery):
                x = x.permute(0, 2, 3, 4, 1).contiguous().view(B, -1, C // 2)
        return x


class SwinUnet(nn.Module):
    """Checkpoint-compatible encoder-decoder for packed One2N fields."""

    def __init__(
        self,
        *,
        img_size,
        patch_size,
        in_chans,
        embed_dim,
        depths,
        updown_sizes,
        num_heads,
        window_size,
        mlp_ratio,
        qkv_bias,
        qk_scale,
        drop_rate,
        attn_drop_rate,
        drop_path_rate,
        ape,
        patch_norm,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        img_size = np.asarray(img_size, dtype=int)
        patch_size = np.asarray(patch_size, dtype=int)
        window_size = np.asarray(window_size, dtype=int)
        updown_sizes = np.asarray(updown_sizes, dtype=int)
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.patch_recovery = PatchRecovery(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            out_chans=in_chans,
            norm_layer=None,
        )
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        current_resolution = patches_resolution
        for i_layer in range(self.num_layers // 2):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=current_resolution,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                updownsample=PatchEmbed if i_layer < self.num_layers // 2 - 1 else None,
                updown_size=(updown_sizes[i_layer] if i_layer < self.num_layers // 2 - 1 else None),
            )
            self.layers.append(layer)
            if i_layer < self.num_layers // 2 - 1:
                current_resolution = current_resolution // updown_sizes[i_layer]
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers // 2, self.num_layers):
            concat_linear = nn.Linear(
                2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
            )
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** (self.num_layers - i_layer - 1)),
                input_resolution=current_resolution,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                updownsample=PatchRecovery if i_layer < self.num_layers - 1 else None,
                updown_size=(
                    updown_sizes[self.num_layers - i_layer - 2]
                    if i_layer < self.num_layers - 1
                    else None
                ),
            )
            self.layers_up.append(layer)
            self.concat_back_dim.append(concat_linear)
            if i_layer < self.num_layers - 1:
                current_resolution = (
                    current_resolution * updown_sizes[self.num_layers - i_layer - 2]
                )
        # Retained with concat_back_dim[0] for strict original-checkpoint loading.
        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_down(self, x):
        x_downsample = []
        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)
        return (x, x_downsample)

    def forward_up(self, x, x_downsample):
        num_layers = self.num_layers // 2 - 1
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[num_layers - inx]], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)
        return x

    def forward(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x, x_downsample = self.forward_down(x)
        x = self.forward_up(x, x_downsample)
        x = self.patch_recovery(x)
        return x
