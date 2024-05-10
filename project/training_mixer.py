import os, sys

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

import argparse
from tqdm import tqdm
import numpy as np
import torch
from torchvision import utils as vutils
from torchvision.transforms import GaussianBlur
from torch.optim.lr_scheduler import StepLR
from model.discriminator import Discriminator
from model.mixer import Mixer
from utils.lpips import LPIPS
import utils.eval as eval
from utils.utils import load_davisset

class TrainMixer:
    
    def __init__(self, args):
        # self.mixer = Mixer(args).to(device=args.device)
        self.mixer = self.load_mixer(args)
        self.discriminator = self.load_discriminator(args)
        self.perceptual_loss = LPIPS().eval().to(device=args.device)
        self.opt = self.configure_optimizers(args)
        self.scheduler = StepLR(self.opt, step_size=1, gamma=args.step_rate)
        
        self.prepare_training()
        print("Mixer Initialized")
        self.train(args)
    
    def configure_optimizers(self, args):
        opt = torch.optim.Adam(
            self.mixer.parameters(),
            lr=args.learning_rate, eps=1e-08, betas=(args.beta1, args.beta2),
            weight_decay=1e-5)
        return opt

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
    def load_mixer(args):
        model = Mixer(args).to(args.device)
        if args.load:
            model.load_checkpoint(args.mix_checkpoint_path)
        return model

    @staticmethod
    def detailed(args, event_frame: torch.Tensor, imgs, decoded_images):
        rec = (imgs - decoded_images) ** 2
        mask = torch.tan(1.5 * event_frame.abs())
        mask = GaussianBlur(39)(mask)
        vutils.save_image(mask, os.path.join("results/mask/mask2.jpg"))

        loss = torch.mul(rec, mask) * args.hotarea_factor + rec
        return loss.mean() / (args.hotarea_factor + 1)

    def train(self, args):
        train_dataset = load_davisset(args)
        for epoch in range(args.epochs):
            self.scheduler.step()

            with tqdm(range(len(train_dataset))) as pbar:
                self.opt.zero_grad()

                for accu_step, imgs in zip(pbar, train_dataset):    
                    imgs = imgs.to(device=args.device)
                    event_frames = imgs[:-1][:,1:]
                    decoded_images, _ = self.mixer(imgs[:-1])
                    imgs = imgs[1:][:,0:1]

                    disc_fake = self.discriminator(decoded_images)

                    loss = eval.tensor_eval(imgs, decoded_images, self.perceptual_loss)
                    perceptual_loss = loss[3]
                    g_loss = -torch.mean(disc_fake)
                    λ = self.mixer.vqgan.calculate_lambda(perceptual_loss, g_loss)
                    
                    gan_loss = args.gan_factor * λ * g_loss
                    con_loss = (args.lpips_factor * perceptual_loss +\
                         self.detailed(args, event_frames, imgs, decoded_images)) * args.con_factor

                    tot_loss = (gan_loss + con_loss) / args.accu_times
                    tot_loss.backward()

                    if accu_step % args.accu_times == 0:
                        self.opt.step()
                        self.opt.zero_grad()
                    
                    real_fake_images = torch.cat((imgs[:4].add(1).mul(0.5), event_frames[:4].add(1).mul(0.5), decoded_images[:4].add(1).mul(0.5)))
                    vutils.save_image(real_fake_images, os.path.join("results/mixer.jpg"), nrow=4)
                    
                    if accu_step % 10 == 0:
                        with torch.no_grad():
                            real_fake_images = torch.cat((imgs[:4].add(1).mul(0.5), event_frames[:4].add(1).mul(0.5), decoded_images[:4].add(1).mul(0.5)))
                            os.makedirs("results/mixer/" + args.dataset + "/" + str(epoch), exist_ok=True)
                            vutils.save_image(real_fake_images, os.path.join("results/mixer", args.dataset, str(epoch), f"{epoch}_{accu_step}.jpg"), nrow=4)

                    pbar.set_postfix(
                        Epoch=epoch,
                        GAN=np.round(gan_loss.cpu().detach().numpy().item(), 5),
                        MSE=np.round(loss[0].cpu().detach().numpy().item(), 5),
                        PSNR=np.round(loss[1].cpu().detach().numpy().item(), 5),
                        SSIM=np.round(loss[2].cpu().detach().numpy().item(), 5),
                        LPIPS=np.round(loss[3].cpu().detach().numpy().item(), 5)
                    )
                    pbar.update(0)
                os.makedirs("checkpoints/mixer/" + args.dataset, exist_ok=True)
                torch.save(self.mixer.state_dict(), os.path.join("checkpoints/mixer", args.dataset, f"epoch_{epoch}.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MIXER")
    parser.add_argument('--latent-dim', type=int, default=400, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=400, help='Image height and width (default: 256)')
    parser.add_argument('--split', type=bool, default=False)
    parser.add_argument('--codebook-size', type=int, default=512, help='Number of codebook vectors (default: 1024)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=1, help='Number of channels of images (default: 1)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=5, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 100)')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--beta1', type=float, default=0.3, help='Adam beta param (default: 0.5)')
    parser.add_argument('--beta2', type=float, default=0.7, help='Adam beta param (default: 0.999)')
    parser.add_argument('--gan-factor', type=float, default=3, help='')
    parser.add_argument('--con-factor', type=float, default=5., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--lpips-factor', type=float, default=1., help='Weighting factor for perceptual loss.')
    parser.add_argument('--hotarea-factor', type=float, default=3., help='')
    parser.add_argument('--accu-times', type=int, default=5, help='Times of gradient accumulation.')
    parser.add_argument('--vqg-checkpoint-path', type=str)
    parser.add_argument('--dis-checkpoint-path', type=str)
    parser.add_argument('--mix-checkpoint-path', type=str)
    parser.add_argument('--load', type=bool, default=True)
    parser.add_argument('--step-rate', type=float, default=0.8)
    parser.add_argument('--sam', type=bool, default=False)

    args = parser.parse_args()
    args.dataset = "1"
    args.dataset_format = 'aedat'
    args.vqg_checkpoint_path = "checkpoints/vqgan/"+args.dataset+"/epoch_15.pt"
    args.dis_checkpoint_path = "checkpoints/discriminator/"+args.dataset+"/epoch_15.pt"
    args.mix_checkpoint_path = "checkpoints/mixer/"+args.dataset+"/epoch_8m.pt"
    
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
        print("PyTorch is using GPU")
    else:
        args.device = torch.device("cpu")
        print("PyTorch is using CPU")
    
    train_mixer = TrainMixer(args)



