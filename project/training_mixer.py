import os, sys

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

import argparse
from tqdm import tqdm
import numpy as np
import torch
from torchvision import utils as vutils
from torch.optim.lr_scheduler import StepLR
from model.discriminator import Discriminator
from model.mixer import Mixer
from utils.lpips import LPIPS
from utils.utils import load_davisset

class TrainMixer:
    
    def __init__(self, args):
        self.mixer = Mixer(args).to(device=args.device)
        self.discriminator = self.load_discriminator(args)
        self.perceptual_loss = LPIPS().eval().to(device=args.device)
        self.opt_con, self.opt_bar = self.configure_optimizers(args)
        self.scheduler_con = StepLR(self.opt_con, step_size=3, gamma=0.5)
        self.scheduler_bar = StepLR(self.opt_bar, step_size=3, gamma=0.5)

        self.prepare_training()
        print("Mixer Initialized")
        self.train(args)
    
    def configure_optimizers(self, args):
        con_lr = args.con_learning_rate
        bar_lr = args.bar_learning_rate
        opt_con = torch.optim.Adam(
            self.mixer.parameters(),
            lr=con_lr, eps=1e-08, betas=(args.beta1, args.beta2))
        opt_bar = torch.optim.Adam(
            self.mixer.parameters(),
            lr=bar_lr, eps=1e-08, betas=(args.beta1, args.beta2))

        return opt_con, opt_bar

    @staticmethod
    def prepare_training():
        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)
    
    @staticmethod
    def load_discriminator(args):
        model = Discriminator(args).to(args.device)
        model.load_checkpoint(args.dis_checkpoint_path)
        model = model.eval()
        return model

    @staticmethod
    def detailed(args, event_frame: torch.Tensor, rec: torch.Tensor):
        event_frame = event_frame.abs()
        mask = event_frame
        # k_size = 7
        # pad = k_size // 2
        # for i in range(pad, event_frame.shape[0]-pad):sss
        #     for j in range(pad, event_frame.shape[1]-pad):
        #         kernel = np.ones((k_size, k_size))
        #         tmp = event_frame[i-pad:i+pad+1, j-pad:j+pad+1]
        #         tmp = np.multiply(kernel, tmp)
        #         mask[i][j] = np.sum(tmp) * 5
        mask = torch.Tensor(mask).to(args.device).detach()
        # mask = torch.atan(mask)
        # vutils.save_image(mask, os.path.join("results/mask/mask.jpg"))
        loss = torch.mul(rec, mask)
        return loss.mean()

    def train(self, args):
        train_dataset = load_davisset(args)
        for epoch in range(args.epochs):
            self.scheduler_con.step()
            self.scheduler_bar.step()

            with tqdm(range(len(train_dataset))) as pbar:
                self.opt_con.zero_grad()
                self.opt_bar.zero_grad()

                for accu_step, imgs in zip(pbar, train_dataset):    
                    imgs = imgs.to(device=args.device)
                    event_frames = imgs[:-1][:,1:]
                    decoded_images, q_loss = self.mixer(imgs[:-1])
                    imgs = imgs[1:][:,0:1]

                    disc_fake = self.discriminator(decoded_images)

                    perceptual_loss = self.perceptual_loss(imgs, decoded_images)
                    rec_loss = torch.abs(imgs - decoded_images)
                    perceptual_loss = args.perceptual_loss_factor * perceptual_loss
                    perceptual_loss = perceptual_loss.mean()
                    g_loss = -torch.mean(disc_fake)
                    bar_loss = self.detailed(args, event_frames, args.rec_loss_factor * rec_loss)
                    λ = self.mixer.vqgan.calculate_lambda(perceptual_loss, g_loss)
                    con_loss = perceptual_loss + args.disc_factor * λ * g_loss + q_loss

                    con_loss = con_loss / args.accu_times
                    con_loss.backward(retain_graph=True)

                    bar_loss = bar_loss / args.accu_times
                    bar_loss.backward()

                    if accu_step % args.accu_times == 0:
                        self.opt_con.step()
                        self.opt_bar.step()
                        self.opt_con.zero_grad()
                        self.opt_bar.zero_grad()

                    if accu_step % 10 == 0:
                        with torch.no_grad():
                            real_fake_images = torch.cat((imgs[:4].add(1).mul(0.5), decoded_images[:4].add(1).mul(0.5)))
                            os.makedirs("results/mixer/" + args.dataset + "/" + str(epoch), exist_ok=True)
                            vutils.save_image(real_fake_images, os.path.join("results/mixer", args.dataset, str(epoch), f"{epoch}_{accu_step}.jpg"), nrow=4)

                    pbar.set_postfix(
                        Epoch=epoch,
                        CON_Loss=np.round(con_loss.cpu().detach().numpy().item(), 5),
                        BAR_Loss=np.round(bar_loss.cpu().detach().numpy().item(), 5)
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
    parser.add_argument('--con-learning-rate', type=float, default=1e-04, help='Learning rate.')
    parser.add_argument('--bar-learning-rate', type=float, default=2e-05, help='Learning rate.')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.5)')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam beta param (default: 0.999)')
    # parser.add_argument('--disc-start', type=int, default=2000, help='When to start the discriminator (default: 10000)')
    parser.add_argument('--disc-factor', type=float, default=2, help='')
    parser.add_argument('--rec-loss-factor', type=float, default=4, help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=2, help='Weighting factor for perceptual loss.')
    parser.add_argument('--bar-factor', type=float, default=1.5, help='')
    parser.add_argument('--accu-times', type=int, default=2, help='Times of gradient accumulation.')
    parser.add_argument('--vqg-checkpoint-path', type=str)
    parser.add_argument('--dis-checkpoint-path', type=str)

    args = parser.parse_args()
    args.dataset = "1"
    args.dataset_format = 'aedat'
    args.split = False
    args.vqg_checkpoint_path = "checkpoints/vqgan/"+args.dataset+"/epoch_15.pt"
    args.dis_checkpoint_path = "checkpoints/discriminator/"+args.dataset+"/epoch_15.pt"
    
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
        print("PyTorch is using GPU")
    else:
        args.device = torch.device("cpu")
        print("PyTorch is using CPU")
    
    train_mixer = TrainMixer(args)



