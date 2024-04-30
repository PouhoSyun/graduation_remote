import os, sys

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-3])
sys.path.append(root_path)

from project.utils.utils import pack_event_stream, Event_Dataset
from project.utils.unpack import unpack
import torchvision.utils as vutils
import numpy as np
import argparse
import torch
import torch.utils.data as data

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
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 100)')
    parser.add_argument('--learning-rate', type=float, default=1e-02, help='Learning rate.')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.5)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-factor', type=float, default=2, help='')
    parser.add_argument('--rec-loss-factor', type=float, default=2.5, help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1.5, help='Weighting factor for perceptual loss.')
    parser.add_argument('--bar-factor', type=float, default=0., help='')
    parser.add_argument('--accu-times', type=int, default=2, help='Times of gradient accumulation.')
    parser.add_argument('--vqg-checkpoint-path', type=str)
    parser.add_argument('--dis-checkpoint-path', type=str)
    parser.add_argument('--mix-checkpoint-path', type=str)
    parser.add_argument('--load', type=bool, default=False)
    parser.add_argument('--step-rate', type=float, default=0.8)

    args = parser.parse_args()
    args.dataset = "1"
    args.dataset_format = 'aedat'
    args.step_rate = 0.8
    args.split = False
    args.load = False
    args.vqg_checkpoint_path = "checkpoints/vqgan/"+args.dataset+"/epoch_15.pt"
    args.dis_checkpoint_path = "checkpoints/discriminator/"+args.dataset+"/epoch_15.pt"
    args.mix_checkpoint_path = "checkpoints/mixer/"+args.dataset+"/epoch_0.pt"

    events, _ = unpack("1")
    dataset = Event_Dataset(events, 400, False)
    train_loader = data.DataLoader(dataset, batch_size=5, shuffle=False)
    for imgs in train_loader:
        imgs = imgs.to(args.device)
        vutils.save_image(imgs[:4], "results/events.jpg", normalize=True,
                          value_range=(-1, 1))
        pass