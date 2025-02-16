import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.utils.spectral_norm(nn.Conv2d(in_channels, in_channels // 8, 1))
        self.key = nn.utils.spectral_norm(nn.Conv2d(in_channels, in_channels // 8, 1))
        self.value = nn.utils.spectral_norm(nn.Conv2d(in_channels, in_channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1))# Learned weighting factor

    def forward(self, x):
        batch_size, C, H, W = x.shape
        q = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)  # (B, H*W, C//8)
        k = self.key(x).view(batch_size, -1, H * W)  # (B, C//8, H*W)
        v = self.value(x).view(batch_size, -1, H * W)  # (B, C, H*W)

        attn = torch.bmm(q, k)  # (B, H*W, H*W)
        attn = torch.softmax(attn, dim=-1)

        out = torch.bmm(v, attn.permute(0, 2, 1))  # (B, C, H*W)
        out = out.view(batch_size, C, H, W)
        return self.gamma * out + x  # Skip connection

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity(),
        )

    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, img_channels, num_features=64, num_residuals=9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                img_channels,
                num_features,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )
        self.down_blocks = nn.ModuleList([
            ConvBlock(
                num_features, num_features * 2, kernel_size=3, stride=2, padding=1
            ),
            ConvBlock(
                num_features * 2,
                num_features * 4,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        ])
        
        # Split residual blocks into two parts with self-attention in between
        mid_res = num_residuals // 2
        self.res_blocks1 = nn.Sequential(*[ResidualBlock(num_features * 4) for _ in range(mid_res)])
        self.self_attn = SelfAttention(num_features * 4)
        self.res_blocks2 = nn.Sequential(*[ResidualBlock(num_features * 4) for _ in range(num_residuals - mid_res)])

        self.up_blocks = nn.ModuleList([
            ConvBlock(
                num_features * 4,
                num_features * 2,
                down=False,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            ConvBlock(
                num_features * 2,
                num_features * 1,
                down=False,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
        ])

        self.last = nn.Conv2d(
            num_features * 1,
            img_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode="reflect",
        )

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.res_blocks1(x)
        x = self.self_attn(x)  # Apply self-attention
        x = self.res_blocks2(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))