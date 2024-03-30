import numpy as np

def peak_signal_to_noise_ratio(image1: np.ndarray, image2: np.ndarray) -> float:
    mse = np.mean((image1 - image2) ** 2)
    max_pixel = np.max(image1)

    psnr = 10 * np.log10(max_pixel ** 2 / mse)

    return psnr