from PIL import Image
import numpy as np
import cv2

img = Image.open("results/mask/mask2.jpg")
img = np.array(img)
kn = np.ones((5, 5))
er = cv2.erode(img, kn, 5)
er = cv2.dilate(er, kn, 5)
Image.fromarray(er).save("results/sam.jpg")