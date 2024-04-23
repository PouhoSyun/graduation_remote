import os, sys

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

from segment_anything.segment_anything import SamPredictor, sam_model_registry
from PIL import Image
import numpy as np
import cv2

def draw_mask(image, mask_generated) :
    masked_image = image.copy()
    shape = masked_image.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if mask_generated[i][j][2]: masked_image[i][j]=np.array([0,255,0], dtype='uint8')
    masked_image = masked_image.astype(np.uint8)
    return cv2.addWeighted(image, 0.3, masked_image, 0.7, 0)

sam = sam_model_registry["vit_h"](checkpoint="segment_anything/vit_h.pth")
predictor = SamPredictor(sam)
img = np.array(Image.open("d:\Working\code\dataset\Flowers\RGB_frame\image_00274.jpg"))
predictor.set_image(img)
masks, _, _ = predictor.predict(point_coords=np.array([[250, 250]]), point_labels=np.array([1]))
img = draw_mask(img, masks.transpose(1, 2, 0))
Image.fromarray(img).show()