import os, sys

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torchsummary import summary
from torchvision import utils as vutils
from model.discriminator import Discriminator
from model.mixer import Mixer
from utils.lpips import LPIPS
from utils.utils import load_davisset, weights_init

class TrainMixer:
    def __init__(self, args):
        self.mixer = Mixer(args).to(device=args.device)
        self.discriminator = Discriminator(args).to(device=args.device)
        self.discriminator.apply(weights_init)
        self.perceptual_loss = LPIPS().eval().to(device=args.device)
        self.opt_con, self.opt_disc = self.configure_optimizers(args)

        self.prepare_training()
        # summary(self.mixer, (2, 400, 400))
        # print(self.mixer)
        self.train(args)
    
    def configure_optimizers(self, args):
        con_lr = args.con_learning_rate
        gan_lr = args.gan_learning_rate
        opt_con = torch.optim.Adam(
            self.mixer.parameters(),
            lr=con_lr, eps=1e-08, betas=(args.beta1, args.beta2))
        opt_disc = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=gan_lr, eps=1e-08, betas=(args.beta1, args.beta2))

        return opt_con, opt_disc

    @staticmethod
    def prepare_training():
        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

    def train(self, args):
        train_dataset = load_davisset(args)
        steps_per_epoch = len(train_dataset)
        for epoch in range(args.epochs):
            with tqdm(range(len(train_dataset))) as pbar:
                self.opt_con.zero_grad()
                self.opt_disc.zero_grad()

                for accu_step, imgs in zip(pbar, train_dataset):
                    imgs = imgs.to(device=args.device)
                    decoded_images, q_loss = self.mixer(imgs[:-1])
                    imgs = imgs[1:][:,0:1]

                    disc_real = self.discriminator(imgs)
                    disc_fake = self.discriminator(decoded_images)

                    disc_factor = self.mixer.vqgan.adopt_weight(args.disc_factor, i=epoch*steps_per_epoch+accu_step, threshold=args.disc_start)

                    perceptual_loss = self.perceptual_loss(imgs, decoded_images)
                    rec_loss = torch.abs(imgs - decoded_images)
                    perceptual_rec_loss = args.perceptual_loss_factor * perceptual_loss + args.rec_loss_factor * rec_loss
                    perceptual_rec_loss = perceptual_rec_loss.mean()
                    g_loss = -torch.mean(disc_fake)

                    λ = self.mixer.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
                    con_loss = perceptual_rec_loss + q_loss + disc_factor * λ * g_loss

                    d_loss_real = torch.mean(F.relu(1. - disc_real))
                    d_loss_fake = torch.mean(F.relu(1. + disc_fake))
                    gan_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)

                    con_loss = con_loss / args.accu_times
                    con_loss.backward(retain_graph=True)

                    gan_loss = gan_loss / args.accu_times
                    gan_loss.backward()

                    if accu_step % args.accu_times == 0:
                        self.opt_con.step()
                        self.opt_disc.step()
                        self.opt_con.zero_grad()
                        self.opt_disc.zero_grad()

                    if accu_step % 10 == 0:
                        with torch.no_grad():
                            real_fake_images = torch.cat((imgs[:4].add(1).mul(0.5), decoded_images[:4].add(1).mul(0.5)))
                            os.makedirs("results/mixer/" + args.dataset + "/" + str(epoch), exist_ok=True)
                            vutils.save_image(real_fake_images, os.path.join("results/mixer", args.dataset, str(epoch), f"{epoch}_{accu_step}.jpg"), nrow=4)

                    pbar.set_postfix(
                        Epoch=epoch,
                        CON_Loss=np.round(con_loss.cpu().detach().numpy().item(), 5),
                        GAN_Loss=np.round(gan_loss.cpu().detach().numpy().item(), 3)
                    )
                    pbar.update(0)
                os.makedirs("checkpoints/mixer/" + args.dataset, exist_ok=True)
                torch.save(self.mixer.vqgan.state_dict(), os.path.join("checkpoints/mixer", args.dataset, f"epoch_{epoch}.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MIXER")
    parser.add_argument('--latent-dim', type=int, default=400, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=400, help='Image height and width (default: 256)')
    parser.add_argument('--split', type=bool, default=True)
    parser.add_argument('--codebook-size', type=int, default=512, help='Number of codebook vectors (default: 1024)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=1, help='Number of channels of images (default: 1)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=5, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train (default: 100)')
    parser.add_argument('--con-learning-rate', type=float, default=1e-05, help='Learning rate.')
    parser.add_argument('--gan-learning-rate', type=float, default=5e-06, help='Learning rate.')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.5)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=1600, help='When to start the discriminator (default: 10000)')
    parser.add_argument('--disc-factor', type=float, default=1, help='')
    parser.add_argument('--rec-loss-factor', type=float, default=3, help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1.5, help='Weighting factor for perceptual loss.')
    parser.add_argument('--accu-times', type=int, default=2, help='Times of gradient accumulation.')
    parser.add_argument('--checkpoint-path', type=str, default=r"./checkpoints/vqgan/1/epoch_49.pt")

    args = parser.parse_args()
    args.dataset = "1"
    args.dataset_format = 'aedat'
    args.split = False
    args.checkpoint_path = r"./checkpoints/vqgan/"+args.dataset+r"/epoch_49.pt"
    
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
        print("PyTorch is using GPU")
    else:
        args.device = torch.device("cpu")
        print("PyTorch is using CPU")
    
    train_mixer = TrainMixer(args)



