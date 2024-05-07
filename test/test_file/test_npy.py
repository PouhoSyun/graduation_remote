import numpy as np
import os, cv2
import matplotlib.pyplot as plt

path = "results/sam1"
filename = os.listdir(path)
cor = []
for name in filename:
    pth = path + '/' + name
    img = cv2.imread(pth, cv2.IMREAD_COLOR)[:,-346:]
    corr = np.where(img[:,:,1] > img[:,:,0], 255, 0)
    # cv2.imwrite("results/sam.jpg",corr)
    cor.append(corr.sum() / 255)

plt.plot(range(1,len(cor)+1),cor)
plt.show()
plt.pause(0)