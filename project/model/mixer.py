import torch
import torch.nn as nn
from model.vqgan import VQGAN
import utils.methods as methods

class WeightedSumLayer(nn.Module):
    def __init__(self, args):
        super(WeightedSumLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(2))

    def forward(self, inputs):
        img1 = inputs[:, 0:1]
        img2 = inputs[:, 1:]
        weighted_sum = self.weight[0] * img1 + self.weight[1] * img2
        return weighted_sum
    
    def load_checkpoint(self, path):
        data = torch.load(path)
        self.load_state_dict(data)
    
class MixModel(nn.Module):
    def __init__(self, args):
        super(MixModel, self).__init__()
        latent_dim = args.latent_dim

        channels = [64, 64, 64, 128, 128, 256]
        attn_resolutions = [25]
        res_block_cnt = 2
        resolution = 400
        layers = [nn.Conv2d(2, 8, 3, 1, 1),
                nn.BatchNorm2d(num_features=8),
                nn.LeakyReLU(0.2),
                nn.Conv2d(8, 8, 3, 1, 1),
                nn.BatchNorm2d(num_features=8),
                nn.LeakyReLU(0.2),
                nn.Conv2d(8, 24, 3, 1, 1),
                nn.BatchNorm2d(num_features=24),
                nn.LeakyReLU(0.2),
                nn.Conv2d(24, channels[0], 3, 1, 1)]
        
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
        layers.append(nn.Conv2d(channels[-1], latent_dim, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def load_checkpoint(self, path):
        data = torch.load(path)
        self.load_state_dict(data)

    def forward(self, x):
        return self.model(x)

class Mixer(nn.Module):
    def __init__(self, args):
        super(Mixer, self).__init__()
        # self.vqgan = self.load_vqgan(args)
        # self.model = self.load_mixmodel(args)
        self.vqgan = VQGAN(args)
        self.model = MixModel(args)
        self.latent_dim = args.latent_dim
    
    @staticmethod
    def load_vqgan(args):
        model = VQGAN(args).to(args.device)
        model.load_checkpoint(args.vqg_checkpoint_path)
        model.encoder.eval()
        model.quant_conv.eval()
        model.codebook.eval()
        model.post_quant_conv.eval()
        model.decoder.eval()
        return model
    
    @staticmethod
    def load_mixmodel(args):
        # model = WeightedSumLayer(args).to(args.device)
        model = MixModel(args).to(args.device)
        if args.load:
            model.load_checkpoint(args.mix_checkpoint_path)
        return model

    def encode_to_z(self, x):
        # encoded_images, _, _ = self.vqgan.encode(self.model(x))
        encoded_images = self.model(x)
        quant_conv_encoded_images = self.vqgan.quant_conv(encoded_images)
        quant_z, indices, q_loss = self.vqgan.codebook(quant_conv_encoded_images)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices, q_loss

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

    def load_checkpoint(self, path):
        data = torch.load(path)
        self.load_state_dict(data)