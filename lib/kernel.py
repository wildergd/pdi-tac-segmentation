import numpy as np
from typing import Any

def create_gaussian_kernel(size: int = 3, sig: int = 1) -> np.ndarray:
    if size % 2 == 0:
        raise ValueError('Kernel size must be odd')
    ax = np.linspace(- size // 2, size // 2, size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def create_laplacian_kernel(alpha: float) -> np.ndarray:
    return np.array([[alpha, 1-alpha, alpha], [1 - alpha, -4, 1 - alpha], [alpha, 1-alpha, alpha]]) * 1 / (1 + alpha)

def create_log_kernel(size: int, alpha: float) -> np.ndarray:
    LoG  = lambda x, y, alpha: (-1 / np.pi * alpha) * (1 - (x**2 + y**2) / 2 * alpha **2) * np.exp(-(x**2 + y**2) / 2 * alpha **2)
    return np.array([LoG(x, y, alpha) for x in range(size) for y in range(size)]).reshape((size, size))

def create_custom_kernel(size: int = 3, value: Any = None) -> np.ndarray:
    if size % 2 == 0:
        raise ValueError('Kernel size must be odd')

    if value is None:
        kernel = np.zeros((size, size), np.float32)
        kernel[size // 2][size // 2] = 1
        return kernel

    if isinstance(value,(list, tuple, np.ndarray)):
        tmp_value = np.array(value)
        if tmp_value.ndim == 1:
            tmp_value = tmp_value.reshape((size, size))
        
        rows, cols = tmp_value.shape
        if rows != cols:
            raise ValueError('Kernel must be squared')

        if rows % 2 == 0:
            raise ValueError('Kernel size must be odd')
        
        return tmp_value
        
    return np.ones((size, size), np.float32) * value