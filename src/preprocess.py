import cv2
import numpy as np


def load_hand(path, size=512):
    """Load an X-ray, pad it to a square with black bars, resize to size x size."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    h, w = img.shape
    side = max(h, w)

    # Put the picture in the middle of a black square canvas
    canvas = np.zeros((side, side), dtype=img.dtype)
    top = (side - h) // 2
    left = (side - w) // 2
    canvas[top:top + h, left:left + w] = img

    # Shrink the square to size x size
    return cv2.resize(canvas, (size, size))