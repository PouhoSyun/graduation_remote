import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupNorm(nn.Module):
    def __init__(self, channel_cnt):
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(num_groups=16, num_channels=channel_cnt, eps=1e-6, affine=True)

    def forward(self, x):
        return self.gn(x)
    
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# residual block for resnet structure, learn to make loss function self.block(x) -> 0
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            GroupNorm(in_channels),
            Swish(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            GroupNorm(out_channels),
            Swish(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )

        if in_channels != out_channels:
            self.channel_up = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        if(self.in_channels != self.out_channels):
            return self.channel_up(x) + self.block(x)
        else:
            return x + self.block(x)
        
class UpSampleBlock(nn.Module):
    def __init__(self, channels):
        super(UpSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0)
        return self.conv(x)
    
class DownSampleBlock(nn.Module):
    def __init__(self, channels):
        super(DownSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 2, 0)

    def forward(self, x):
        x = F.pad(x, (0, 1, 0, 1), "constant", 0)
        return self.conv(x)

# self attention layer usage -> non-local block module
class NonLocalBlock(nn.Module):
    def __init__(self, channels):
        super(NonLocalBlock, self).__init__()
        self.in_channels = channels
        self.gn = GroupNorm(channels)
        self.q = nn.Conv2d(channels, channels, 1, 1, 0)
        self.k = nn.Conv2d(channels, channels, 1, 1, 0)
        self.v = nn.Conv2d(channels, channels, 1, 1, 0)
        self.proj_out = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        x_norm = self.gn(x)
        q = self.q(x_norm)
        k = self.k(x_norm)
        v = self.v(x_norm)

        b, c, h, w = q.shape

        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w)

        attn = torch.bmm(q, k)
        attn = attn * (int(c) ** (-0.5))
        attn = F.softmax(attn, dim=2)
        attn = attn.permute(0, 2, 1)

        val = torch.bmm(v, attn)
        val = val.reshape(b, c, h, w)

        return x + val