import torch
import torch.nn as nn

# a dictionary of vectors for mapping
class Codebook(nn.Module):
    def __init__(self, args):
        super(Codebook, self).__init__()
        self.codebook_size = args.codebook_size
        self.latent_dim = args.latent_dim
        self.beta = args.beta

        # create initial vector codebook with latent_dim words
        self.embedding = nn.Embedding(self.codebook_size, self.latent_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()
        x_flattened = x.view(-1, self.latent_dim)

        d = torch.sum(x_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - \
            (torch.matmul(x_flattened, self.embedding.weight.t())) * 2
        min_encoding_indices = torch.argmin(d, dim=1)
        x_vq: torch.Tensor = self.embedding(min_encoding_indices).view(x.shape)

        loss = torch.mean((x_vq.detach() - x) ** 2) + self.beta * torch.mean((x_vq - x.detach()) ** 2)
        x_vq = x + (x_vq - x).detach()
        x_vq = x_vq.permute(0, 3, 1, 2)

        return x_vq, min_encoding_indices, loss