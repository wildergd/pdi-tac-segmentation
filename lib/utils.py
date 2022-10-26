import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from .misc import get_image_dimensions
from .colors import fix_image_colors

def imshow(*images, titles = None, cols = 3, titleColor = 'black', cmap = 'gray'):
    rows = ceil(len(images) / cols)
    size = 16 / cols
    plt.figure(figsize=(cols * size, rows * size), frameon=False, layout=None)
    plt.axis('off')
    plt.margins(0)
    plt.subplots_adjust(hspace=0, wspace=0)
    for i in range(len(images)):
        _, _, depth = get_image_dimensions(images[i])
        plt.subplot(rows, cols, i+1)
        if depth == 1:
            plt.imshow(images[i], vmin = images[i].min(), vmax = images[i].max(), cmap = cmap)
        else:
            plt.imshow(fix_image_colors(images[i]))
        if titles is not None:
            plt.title(titles[i], color = titleColor)
        plt.xticks([])
        plt.yticks([])
    plt.show()

def histograms(*images, titles = None, cols = 3, titleColor = 'black'):
    rows = ceil(len(images) / cols)
    row_size = 3
    size = 16 / cols
    plt.figure(figsize=(cols * size, rows * row_size))
    for i in range(len(images)):
        _, _, depth = get_image_dimensions(images[i])
        plt.subplot(rows, cols, i+1)
        if depth == 1:
            histogram, bin_edges = np.histogram(images[i], bins=256, range=(0, 256))
            plt.plot(bin_edges[0:-1], histogram)
        else:
            colors = ('red', 'green', 'blue')
            img = fix_image_colors(images[i])
            for channel in range(depth):
                histogram, bin_edges = np.histogram(img[:, :, channel], bins=256, range=(0, 256))
                plt.plot(bin_edges[0:-1], histogram, color=colors[channel])
        if titles is not None:
            plt.title(titles[i], color = titleColor)
    plt.show()


def plot_histograms(*data, titles = None, cols = 3, titleColor = 'black', bins=10):
    rows = ceil(len(data) / cols)
    row_size = 3
    size = 16 / cols
    plt.figure(figsize=(cols * size, rows * row_size))
    for i in range(len(data)):
        plt.subplot(rows, cols, i+1)
        hist_data = np.array([len(contour) for contour in data[i]])
        histogram, bin_edges = np.histogram(hist_data, bins=hist_data.max(), range=(hist_data.min(), hist_data.max()))
        plt.plot(bin_edges[0:-1], histogram)
        if titles is not None:
            plt.title(titles[i], color = titleColor)
    plt.show()
