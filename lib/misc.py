import numpy as np

get_image_dimensions = lambda img: (*img.shape, 1) if len(img.shape) == 2 else img.shape

def pad_image(image: np.ndarray, size = 1) -> np.ndarray:
    img_height, img_width = image.shape
    img_expanded = np.zeros((size * 2 +img_height, size * 2 + img_width), np.float32)
    img_expanded[size:size + img_height, size:size + img_width] = image
    return img_expanded

def generate_histogram(img: np.ndarray) -> list:
    return [(img == x).sum() for x in range(0, 256)]

def otsu(img: np.ndarray) -> tuple:
    hist = generate_histogram(img)
    best_threshold = 0
    var_intraclass = np.Inf
    p = hist / np.sum(hist)
    for t in range(1, 256):
        g1 = p[0:t]
        g2 = p[t:255]
        i1 = np.arange(t)
        i2 = np.arange(t, 255)
        q1 = np.sum(g1)
        q2 = np.sum(g2)
        prom1 = np.sum(i1 * g1 / q1)
        prom2 = np.sum(i2 * g2 / q2)
        var1 = np.sum(((i1 - prom1)**2) * g1 / q1)
        var2 = np.sum(((i2 - prom2)**2) * g2 / q2)
        var_ic = q1 * var1 + q2 * var2
        if var_ic < var_intraclass:
            var_intraclass = var_ic
            best_threshold = t

    return best_threshold, var_intraclass