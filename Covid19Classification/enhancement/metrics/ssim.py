import numpy as np
from skimage.metrics import structural_similarity as ssim

def structural_similarity_index_measure(image1: np.ndarray, image2: np.ndarray) -> float:
    return ssim(
        image1, 
        image2,
        data_range = image2.max() - image2.min()
    )