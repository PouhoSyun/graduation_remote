import os, sys

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-3])
sys.path.append(root_path)

from segment_anything.segment_anything import SamPredictor, sam_model_registry
from PIL import Image
import numpy as np
from project.utils.utils import load_davisset
import cv2, argparse
from tqdm import tqdm
import torchvision.utils as vutils

def hotarea(event_frame):
    mask = event_frame
    mask = cv2.erode(mask, np.ones((6, 6)), 4)
    _, mask = cv2.threshold(mask, 130, 255, cv2.THRESH_TOZERO)

    top_pixels = np.argsort(mask.ravel())[-50:]
    top_coords = np.unravel_index(top_pixels, mask.shape)
    point = np.mean(top_coords, axis=1).astype(np.uint8)
    return mask, point

def draw_mask(image, mask_generated) :
    masked_image = image.copy()
    shape = masked_image.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if mask_generated[i][j][2]: masked_image[i][j]=np.array([0,255,0], dtype='uint8')
    masked_image = masked_image.astype(np.uint8)
    return cv2.addWeighted(image, 0.3, masked_image, 0.7, 0, dtype=cv2.CV_8U)

def dump(args):
    train_dataset = load_davisset(args)
    sam = sam_model_registry["vit_b"](checkpoint="segment_anything/pth/vit_b.pth")
    predictor = SamPredictor(sam)

    with tqdm(range(len(train_dataset))) as pbar:
        for step, imgs in zip(pbar, train_dataset):
            if not step % 10: 
                img = np.array(imgs[0].add(1).mul(127.5)).astype(np.uint8)
                mask_hr, point = hotarea(img[1])
                img = cv2.cvtColor(img[0:1].transpose((1, 2, 0)), cv2.COLOR_GRAY2RGB)
                # img = np.array(Image.open("results/vqgan/1/15/15_100.jpg"))[70:330, 27:373]
                predictor.set_image(img)
                masks, _, _ = predictor.predict(point_coords=np.array([(point[1], point[0])]), point_labels=np.array([1]))
                mask = draw_mask(img, masks.transpose(1, 2, 0))
                cv2.circle(mask, [point[1], point[0]], 5, (255, 0, 0))
                mask_hr = cv2.cvtColor(mask_hr, cv2.COLOR_GRAY2RGB)

                img = np.hstack([img, mask_hr, mask]).astype(np.uint8)
                Image.fromarray(img).save("results/sam.jpg")
            
            pbar.set_postfix()
            pbar.update(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SAM")
    parser.add_argument('--latent-dim', type=int, default=400, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=400, help='Image height and width (default: 256)')
    parser.add_argument('--split', type=bool, default=False)
    parser.add_argument('--codebook-size', type=int, default=512, help='Number of codebook vectors (default: 1024)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=1, help='Number of channels of images (default: 1)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=1, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 100)')
    parser.add_argument('--learning-rate', type=float, default=5e-3, help='Learning rate.')
    parser.add_argument('--beta1', type=float, default=0.3, help='Adam beta param (default: 0.5)')
    parser.add_argument('--beta2', type=float, default=0.7, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-factor', type=float, default=0.5, help='')
    parser.add_argument('--rec-loss-factor', type=float, default=8., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=0.5, help='Weighting factor for perceptual loss.')
    parser.add_argument('--bar-factor', type=float, default=2., help='')
    parser.add_argument('--accu-times', type=int, default=3, help='Times of gradient accumulation.')
    parser.add_argument('--vqg-checkpoint-path', type=str)
    parser.add_argument('--dis-checkpoint-path', type=str)
    parser.add_argument('--mix-checkpoint-path', type=str)
    parser.add_argument('--load', type=bool, default=True)
    parser.add_argument('--step-rate', type=float, default=0.9)
    parser.add_argument('--sam', type=bool, default=True)

    args = parser.parse_args()
    args.dataset = "1"
    args.dataset_format = 'aedat'
    args.vqg_checkpoint_path = "checkpoints/vqgan/"+args.dataset+"/epoch_15.pt"
    args.dis_checkpoint_path = "checkpoints/discriminator/"+args.dataset+"/epoch_15.pt"
    args.mix_checkpoint_path = "checkpoints/mixer/"+args.dataset+"/epoch_62m.pt"

    dump(args)
