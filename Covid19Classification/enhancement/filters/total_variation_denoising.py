import numpy as np

from skimage.restoration import denoise_tv_chambolle

def total_variation_denoising(image: np.ndarray, weight: float = 0.1) -> np.ndarray:
    denoising_image = denoise_tv_chambolle(
        image,
        weight = weight
    )

    return denoising_image