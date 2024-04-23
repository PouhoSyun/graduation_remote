import torch.nn as nn
import project.utils.methods as methods

# compress frame to short vector mapped by codebook
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        channels = [64, 64, 64, 128, 128, 256]
        attn_resolutions = [25]
        res_block_cnt = 2
        resolution = 400
        layers = [nn.Conv2d(args.image_channels, channels[0], 3, 1, 1)]
        for i in range(len(channels) - 1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for j in range(res_block_cnt):
                layers.append(methods.ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(methods.NonLocalBlock(in_channels))
            if i != len(channels) - 2:
                layers.append(methods.DownSampleBlock(out_channels))
                resolution //= 2
        
        layers.append(methods.ResidualBlock(channels[-1], channels[-1]))
        layers.append(methods.NonLocalBlock(channels[-1]))
        layers.append(methods.ResidualBlock(channels[-1], channels[-1]))
        layers.append(methods.GroupNorm(channels[-1]))
        layers.append(methods.Swish())
        layers.append(nn.Conv2d(channels[-1], args.latent_dim, 3, 1, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)