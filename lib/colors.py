import numpy as np
import cv2 as cv
from .misc import get_image_dimensions

def fix_image_colors(image):
    w, h, depth = get_image_dimensions(image)
    if depth == 1:
         return image
    b, g, r = cv.split(image)
    return cv.merge([r, g, b])   
