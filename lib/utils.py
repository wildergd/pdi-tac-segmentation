from numpy import min, max
import cv2 as cv
import matplotlib.pyplot as plt
from math import ceil
from .misc import get_image_dimensions
from .colors import fix_image_colors

def imshow(*images, titles=None, cols = 3, titleColor = "black", cmap = 'gray'):
    rows = ceil(len(images) / cols)
    size = 16 / cols
    plt.figure(figsize=(cols * size, rows * size), frameon=False)
    plt.axis('off')
    plt.margins(0)
    plt.subplots_adjust(hspace=0, wspace=0)
    for i in range(len(images)):
        _, _, depth = get_image_dimensions(images[i])
        plt.subplot(rows, cols, i+1)
        if depth == 1:
            plt.imshow(images[i], vmin = min(images[i]), vmax = max(images[i]), cmap = cmap)
        else:
            print()
            plt.imshow(fix_image_colors(images[i]))
        if titles is not None:
            plt.title(titles[i], color = titleColor)
        plt.xticks([])
        plt.yticks([])
    plt.show()
