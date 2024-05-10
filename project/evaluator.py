import os, sys

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

import argparse
import torch
from torchvision import utils as vutils
from model.mixer import Mixer
import utils.eval as eval

class Evaluator:
    
    def __init__(self):
        self.mixer = self.load_mixer(args)
        print("Evaluator Initialized")

    @staticmethod
    def load_mixer(args):
        model = Mixer(args).to(args.device)
        model.load_checkpoint(args.mix_checkpoint_path)
        model = model.eval()
        return model

    def evaluate(self, imgs): 
        imgs = imgs.to(device=args.device)
        decoded_images, _ = self.mixer(imgs)
        event_frames = imgs[:,1:]
        imgs = imgs[:,0:1]
        loss = eval.tensor_eval(imgs, decoded_images, self.perceptual_loss)            
        real_fake_images = torch.cat((imgs[:4].add(1).mul(0.5), event_frames[:4].add(1).mul(0.5), decoded_images[:4].add(1).mul(0.5)))
        vutils.save_image(real_fake_images, os.path.join("results/mixer.jpg"), nrow=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EVALUATOR")
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--mix-checkpoint-path', type=str)
    args = parser.parse_args()
    args.mix_checkpoint_path = "checkpoints/mixer/"+args.dataset+"/epoch_k.pt"
    
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
        print("PyTorch is using GPU")
    else:
        args.device = torch.device("cpu")
        print("PyTorch is using CPU")
    
    evaluator = Evaluator()
    # evaluator.evaluate(imgs)


