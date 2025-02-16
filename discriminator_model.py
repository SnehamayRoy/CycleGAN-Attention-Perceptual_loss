import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.utils.spectral_norm(nn.Conv2d(in_channels, in_channels // 8, 1))
        self.key = nn.utils.spectral_norm(nn.Conv2d(in_channels, in_channels // 8, 1))
        self.value = nn.utils.spectral_norm(nn.Conv2d(in_channels, in_channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.shape
        q = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(batch_size, -1, H * W)
        v = self.value(x).view(batch_size, -1, H * W)

        attn = torch.bmm(q, k)
        attn = torch.softmax(attn, dim=-1)

        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        return self.gamma * out + x  # Skip connection

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                4,
                stride,
                1,
                bias=True,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        in_channels = features[0]
        for idx, feature in enumerate(features[1:]):
            layers.append(
                Block(in_channels, feature, stride=1 if feature == features[-1] else 2)
            )
            in_channels = feature
            # Insert self-attention after second block (before final block)
            if idx == 1:  # After processing 256 channels
                layers.append(SelfAttention(in_channels))

        layers.append(
            nn.Conv2d(
                in_channels,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        return torch.sigmoid(self.model(x))