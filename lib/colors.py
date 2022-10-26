import numpy as np
import cv2 as cv
from .misc import get_image_dimensions, generate_histogram

def fix_image_colors(image):
    _, _, depth = get_image_dimensions(image)
    if depth == 1:
         return image
    b, g, r = cv.split(image)
    return cv.merge([r, g, b])   

def equalize(img: np.ndarray, levels: int = 255) -> np.ndarray:
    cdf = np.cumsum(generate_histogram(img.astype('float')))
    cdf_map = (cdf - np.min(cdf)) * levels / (cdf[-1] - np.min(cdf))
    return cdf_map[img].astype('uint8')

def color_balance(img, low_per, high_per):
    tot_pix = img.shape[1] * img.shape[0]
    low_count = tot_pix * low_per / 100
    high_count = tot_pix * (100 - high_per) / 100
    
    cs_img = []
    for ch in cv.split(img):
        cum_hist_sum = np.cumsum(generate_histogram(ch))

        li, hi = np.searchsorted(cum_hist_sum, (low_count, high_count))
        if (li == hi):
            cs_img.append(ch)
            continue
        lut = np.array([0 if i < li 
                        else (255 if i > hi else round((i - li) / (hi - li) * 255)) 
                        for i in np.arange(0, 256)], dtype = 'uint8')
        cs_ch = cv.LUT(ch, lut)
        cs_img.append(cs_ch)
        
    return cv.merge(cs_img)

def binarize(img: np.ndarray, threshold: float) -> np.ndarray:
    result = np.zeros_like(img)
    result[img < threshold] = 0
    result[img >= threshold] = 1
    return result
