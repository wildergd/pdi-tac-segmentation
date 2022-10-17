import numpy as np
from .misc import pad_image

def custom_transform(image: np.ndarray, kernel: np.ndarray = np.ones((3, 3), np.float32)) -> np.ndarray:
    kern_height, kern_width = kernel.shape
    if kern_height % 2 == 0 or kern_width % 2 == 0:
        raise ValueError('Kernel size must be odd')
    
    extra_cols_rows = kern_width // 2
    img_height, img_width = image.shape
    img_expanded = pad_image(image, extra_cols_rows)
    img_modified = image.copy().astype('float')

    for i in range(img_height):
        for j in range(img_width):
            part = img_expanded[i:i + kern_height, j:j + kern_width]
            img_modified[i, j] = np.sum(part * kernel)

    return img_modified.astype('int')

def median_transform(image: np.ndarray, kern_size:int = 3 ) -> np.ndarray:
    if kern_size % 2 == 0:
        raise ValueError('Kernel size must be odd')
    
    extra_cols_rows = kern_size // 2
    img_height, img_width = image.shape
    img_expanded = pad_image(image, extra_cols_rows)
    img_modified = image.copy().astype('float')

    for i in range(img_height):
        for j in range(img_width):
            part = img_expanded[i:i + kern_size, j:j + kern_size]
            img_modified[i, j] = np.median(part)

    return img_modified.astype('int')

def gaussian_transform(image: np.ndarray, kernel: np.ndarray = np.ones((3, 3), np.float32)) -> np.ndarray:
    kern_height, kern_width = kernel.shape
    if kern_height % 2 == 0 or kern_width % 2 == 0:
        raise ValueError('Kernel size must be odd')
    
    extra_cols_rows = kern_width // 2
    img_height, img_width = image.shape
    img_expanded = pad_image(image, extra_cols_rows)
    img_modified = image.copy().astype('float')

    for i in range(img_height):
        for j in range(img_width):
            part = img_expanded[i:i + kern_height, j:j + kern_width]
            img_modified[i, j] = part[i, j] + kernel

    return img_modified.astype('int')