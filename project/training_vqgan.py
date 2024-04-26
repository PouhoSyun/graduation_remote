import os, sys

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from torch.optim.lr_scheduler import StepLR
from model.discriminator import Discriminator
from utils.lpips import LPIPS
from model.vqgan import VQGAN
from utils.utils import load_frameset, weights_init

class TrainVQGAN:
    def __init__(self, args):
        self.vqgan = VQGAN(args).to(device=args.device)
        # self.vqgan.load_checkpoint("checkpoints/vqgan/"+args.dataset+"/epoch_87.pt")
        self.discriminator = Discriminator(args).to(device=args.device)
        self.discriminator.apply(weights_init)
        # self.discriminator.load_checkpoint("checkpoints/discriminator/"+args.dataset+"/epoch_87.pt")
        self.perceptual_loss = LPIPS().eval().to(device=args.device)
        self.opt_vq, self.opt_disc = self.configure_optimizers(args)
        self.scheduler_vq = StepLR(self.opt_vq, step_size=1, gamma=0.95)
        self.scheduler_disc = StepLR(self.opt_disc, step_size=1, gamma=0.9)

        self.prepare_training()
        self.train(args)

    def configure_optimizers(self, args):
        lr = args.learning_rate
        opt_vq = torch.optim.Adam( 
            list(self.vqgan.encoder.parameters()) +
            list(self.vqgan.decoder.parameters()) +
            list(self.vqgan.codebook.parameters()) +
            list(self.vqgan.quant_conv.parameters()) +
            list(self.vqgan.post_quant_conv.parameters()),
            lr=lr, eps=1e-08, betas=(args.beta1, args.beta2)
        )
        opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                                    lr=lr, eps=1e-08, betas=(args.beta1, args.beta2))

        return opt_vq, opt_disc

    @staticmethod
    def prepare_training():
        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

    def train(self, args):
        train_dataset = load_frameset(args)
        steps_per_epoch = len(train_dataset)
        for epoch in range(args.epochs):
            self.scheduler_vq.step()
            self.scheduler_disc.step()

            with tqdm(range(len(train_dataset))) as pbar:
                self.opt_vq.zero_grad()
                self.opt_disc.zero_grad()

                for accu_step, imgs in zip(pbar, train_dataset):
                    imgs = imgs.to(device=args.device)
                    decoded_images, _, q_loss = self.vqgan(imgs)

                    disc_real = self.discriminator(imgs)
                    disc_fake = self.discriminator(decoded_images)

                    disc_factor = self.vqgan.adopt_weight(args.disc_factor, i=epoch*steps_per_epoch+accu_step, threshold=args.disc_start)

                    perceptual_loss = self.perceptual_loss(imgs, decoded_images)
                    rec_loss = torch.abs(imgs - decoded_images)
                    perceptual_rec_loss = args.perceptual_loss_factor * perceptual_loss + args.rec_loss_factor * rec_loss
                    perceptual_rec_loss = perceptual_rec_loss.mean()
                    g_loss = -torch.mean(disc_fake)

                    λ = self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
                    vq_loss = perceptual_rec_loss + q_loss + disc_factor * λ * g_loss

                    d_loss_real = torch.mean(F.relu(1. - disc_real))
                    d_loss_fake = torch.mean(F.relu(1. + disc_fake))
                    gan_loss = disc_factor * (d_loss_real + d_loss_fake)

                    vq_loss = vq_loss / args.accu_times
                    vq_loss.backward(retain_graph=True)

                    gan_loss = gan_loss / args.accu_times
                    gan_loss.backward()

                    if accu_step % args.accu_times == 0:
                        self.opt_vq.step()
                        self.opt_disc.step()
                        self.opt_vq.zero_grad()
                        self.opt_disc.zero_grad()

                    if accu_step % 10 == 0:
                        with torch.no_grad():
                            real_fake_images = torch.cat((imgs[:4].add(1).mul(0.5), decoded_images[:4].add(1).mul(0.5)))
                            os.makedirs("results/vqgan/" + args.dataset + "/" + str(epoch), exist_ok=True)
                            vutils.save_image(real_fake_images, os.path.join("results/vqgan", args.dataset, str(epoch), f"{epoch}_{accu_step}.jpg"), nrow=4)

                    pbar.set_postfix(
                        Epoch=epoch,
                        VQ_Loss=np.round(vq_loss.cpu().detach().numpy().item(), 5),
                        GAN_Loss=np.round(gan_loss.cpu().detach().numpy().item(), 5)
                    )
                    pbar.update(0)
                os.makedirs("checkpoints/vqgan/" + args.dataset, exist_ok=True)
                os.makedirs("checkpoints/discriminator/" + args.dataset, exist_ok=True)
                torch.save(self.vqgan.state_dict(), os.path.join("checkpoints/vqgan", args.dataset, f"epoch_{epoch}.pt"))
                torch.save(self.discriminator.state_dict(), os.path.join("checkpoints/discriminator", args.dataset, f"epoch_{epoch}.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=400, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=400, help='Image height and width (default: 256)')
    parser.add_argument('--split', type=bool)
    parser.add_argument('--codebook-size', type=int, default=512, help='Number of codebook vectors (default: 1024)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=1, help='Number of channels of images (default: 1)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=4, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 100)')
    parser.add_argument('--learning-rate', type=float, default=1e-03, help='Learning rate (default: 2.25e-05)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.5)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=4000, help='When to start the discriminator (default: 10000)')
    parser.add_argument('--disc-factor', type=float, default=2, help='')
    parser.add_argument('--rec-loss-factor', type=float, default=2.5, help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1.5, help='Weighting factor for perceptual loss.')
    parser.add_argument('--accu-times', type=int, default=2, help='Times of gradient accumulation.')
    args = parser.parse_args()
    args.dataset = "1"
    args.dataset_format = 'aedat'
    args.split = False
    
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:200"
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
        print("PyTorch is using GPU")
    else:
        args.device = torch.device("cpu")
        print("PyTorch is using CPU")
    
    train_vqgan = TrainVQGAN(args)



