import cv2 
import numpy as np

def noiseremoval(img):
    # img = cv2.imread("test_image.png")

    denoise = cv2.fastNlMeansDenoisingColored(img,None,3,3,7,21)

    return denoise
