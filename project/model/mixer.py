import torch
import torch.nn as nn
from model.vqgan import VQGAN
import utils.methods as methods

class Mixer(nn.Module):
    def __init__(self, args):
        super(Mixer, self).__init__()
        self.vqgan = self.load_vqgan(args)
        self.latent_dim = args.latent_dim

        channels = [64, 64, 64, 128, 128, 256]
        attn_resolutions = [25]
        res_block_cnt = 2
        resolution = 400
        layers = [nn.Conv2d(2, 8, 3, 1, 1),
                nn.BatchNorm2d(num_features=8),
                methods.Swish(),
                # nn.Conv2d(8, 8, 3, 1, 1),
                # nn.BatchNorm2d(num_features=8),
                # methods.Swish(),
                nn.Conv2d(8, channels[0], 3, 1, 1)]
        
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

        if args.load:
            self.model.load_state_dict(torch.load(args.mix_checkpoint_path))
            # self.model.eval()
    
    @staticmethod
    def load_vqgan(args):
        model = VQGAN(args).to(args.device)
        model.load_checkpoint(args.vqg_checkpoint_path)
        model = model.eval()
        return model

    def encode_to_z(self, x):
        encoded_images = self.model(x)
        quant_conv_encoded_images = self.vqgan.quant_conv(encoded_images)
        quant_z, indices, q_loss = self.vqgan.codebook(quant_conv_encoded_images)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices, q_loss

    # @torch.no_grad()
    def z_to_image(self, indices, z_size=25):
        ix_to_vectors = self.vqgan.codebook.embedding(indices)
        ix_to_vectors = ix_to_vectors.reshape(indices.shape[0], z_size, z_size, self.latent_dim)
        ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2)
        image = self.vqgan.decode(ix_to_vectors)
        return image
    
    def forward(self, x):
        _, indices, q_loss = self.encode_to_z(x)
        img = self.z_to_image(indices)
        return img, q_loss