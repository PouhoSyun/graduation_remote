import numpy as np
import os, cv2
import matplotlib.pyplot as plt

path = "results/sam1"
filename = os.listdir(path)
cor = []
for name in filename:
    pth = path + '/' + name
    img = cv2.imread(pth, cv2.IMREAD_COLOR)[:,-346:]
    img1 = cv2.imread(pth, cv2.IMREAD_COLOR)[:,-692:-346]
    corr = np.where(img[:,:,1] > img[:,:,0], 255, 0)
    corr1 = np.where(img1[:,:,1] > img1[:,:,0], 255, 0)
    cor.append((corr.sum()-corr1.sum()) / 255)
cor = np.array(cor)
plt.plot(range(1,len(cor)+1),cor/2500,'.')
plt.xlabel("Frames")
plt.ylabel("Segmentation size")
ad = np.count_nonzero(np.where(cor<cor.mean(),1,0))/824
print(ad)
plt.savefig("results/boxplot01",dpi=300)
plt.pause(0)