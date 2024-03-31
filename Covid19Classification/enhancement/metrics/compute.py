from .rmse import root_mean_squared_error
from .ambe import absolute_mean_brightness_error
from .psnr import peak_signal_to_noise_ratio
from .ssim import structural_similarity_index_measure

def compute_enhancement_metrics(image1, image2):
    return {
        "RMSE": root_mean_squared_error(image1, image2),
        "AMBE": absolute_mean_brightness_error(image1, image2),
        "PSNR": peak_signal_to_noise_ratio(image1, image2),
        "SSIM": structural_similarity_index_measure(image1, image2)
    }