from skimage.metrics import structural_similarity as ssim
import numpy as np
import lpips
import cv2, os, torch

def eval(x:torch.Tensor, y:torch.Tensor, loss_fn):
    mse = torch.nn.functional.mse_loss(y, x).reshape(1)
    psnr_value = -10 * torch.log10(mse).reshape(1)

    x_np = x.permute(1, 2, 0).cpu().detach().numpy()
    y_np = y.permute(1, 2, 0).cpu().detach().numpy()
    ssim_value = torch.tensor(ssim(x_np, y_np, channel_axis=-1, data_range=1.0)).reshape(1).to('cuda')

    lpips_value = loss_fn(x, y).reshape(1)
    return torch.cat((mse, psnr_value, ssim_value, lpips_value))

def tensor_eval(ts1:torch.Tensor, ts2:torch.Tensor, loss_fn):
    eva = torch.zeros((1, 4))
    for i in range(ts1.shape[0]):
        t1 = ts1[i].add(1).mul(0.5)
        t2 = ts2[i].add(1).mul(0.5)
        eva = eval(t1, t2, loss_fn)
    return eva / ts1.shape[0]

def group_eval(pth1, pth2):
    fn1 = os.listdir(pth1)
    fn2 = os.listdir(pth2)
    eval = torch.zeros((1, 4))
    for i in range(len(fn1)):
        image1 = cv2.imread(fn1[i])
        image2 = cv2.imread(fn2[i])
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB).astype(np.float32)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB).astype(np.float32)
        image1 = torch.Tensor(image1 / 127.5 - 1)
        image2 = torch.Tensor(image2 / 127.5 - 1)
        eval += eval(image1, image2)
    eval /= len(fn1)
    return eval

if __name__ == '__main__':
    x = torch.rand(4, 1, 400, 400).to('cuda')
    y = torch.rand(4, 1, 400, 400).to('cuda')
    print(tensor_eval(x, y))