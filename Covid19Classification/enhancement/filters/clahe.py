import cv2
import numpy as np

clahe_config = {
    "clipLimit": 5,
    "tileGridSize": (8, 8)
}

def clahe_filter(image: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit = clahe_config["clipLimit"],
        tileGridSize = clahe_config["tileGridSize"]
    )

    return clahe.apply(image)