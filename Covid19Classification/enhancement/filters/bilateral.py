import cv2
import numpy as np

bilateral_config = {
    "diameter": 5,
    "sigmaColor": 75,
    "sigmaSpace": 75
}


def bilateral_filter(image: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(
        image,
        d=bilateral_config["diameter"],
        sigmaSpace=bilateral_config["sigmaSpace"],
        sigmaColor=bilateral_config["sigmaColor"]
    )
