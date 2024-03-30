import numpy as np

def absolute_mean_brightness_error(image1: np.ndarray, image2: np.ndarray) -> float:
    i1 = image1.astype(np.float32)
    i2 = image2.astype(np.float32)

    absolute_errors = np.abs(i1 - i2)

    n_pixels = image1.shape[0] * image1.shape[1]

    return np.sum(absolute_errors) / n_pixels