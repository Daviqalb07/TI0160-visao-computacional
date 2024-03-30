import numpy as np

def root_mean_squared_error(image1: np.ndarray, image2: np.ndarray) -> float:
    squared_difference = (image1 - image2) ** 2

    n_pixels = image1.shape[0] * image1.shape[1]

    return np.sqrt(np.sum(squared_difference) / n_pixels)