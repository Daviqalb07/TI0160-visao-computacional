import cv2
import numpy as np

def histogram_equalization(image: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(image)