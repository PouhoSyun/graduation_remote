import cv2
import numpy as np

def blur_image(image_path, blur_strength=15):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)
    image = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)
    image = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)
    return image

def adjust_brightness(image_path, brightness_value=10):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    brightened_image = cv2.add(image, np.array([brightness_value]))
    return brightened_image

# 使用示例
# 假设我们有一个名为 'example.jpg' 的图像文件
# blurred_img = blur_image('example.jpg', blur_strength=7)
# brightened_img = adjust_brightness('example.jpg', brightness_value=30)

# cv2.imwrite('blurred_example.jpg', blurred_img)
# cv2.imwrite('brightened_example.jpg', brightened_img)

path = "results/output.jpg"
pic = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
pic_bri = adjust_brightness(path)
pic_pep = blur_image(path)
pic = np.vstack([pic, pic_bri, pic_pep])
cv2.imwrite("results/res.jpg", pic)