from skimage.metrics import structural_similarity as ssim
import numpy as np
import lpips
import cv2, os
from math import log10, sqrt

def eval(pth1, pth2):
    image1 = cv2.imread(pth1)
    image2 = cv2.imread(pth2)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        psnr = 100
    else:
        max_pixel = 255.0
        psnr = 20 * log10(max_pixel / sqrt(mse))

    ssim_value = ssim(image1, image2, multichannel=True)

    lpips_metric = lpips.LPIPS(net='alex')
    lpips_value = lpips_metric(image1, image2).item()

    return np.array([psnr, ssim_value, lpips_value])

def group_eval(pth1, pth2):
    fn1 = os.listdir(pth1)
    fn2 = os.listdir(pth2)
    val = np.array([0, 0, 0])
    for i in range(len(fn1)):
        val += eval(fn1[i], fn2[i])
    val /= len(fn1)
    return val