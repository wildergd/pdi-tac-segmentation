get_image_dimensions = lambda img: (*img.shape, 1) if len(img.shape) == 2 else img.shape
