"""EME — Enhancement Measure Estimation (no-reference).

Reference:
    Agaian, S. S., Silver, B., & Panetta, K. A. (2007).
    Transform Coefficient Histogram-Based Image Enhancement Algorithms
    Using Contrast Entropy.
    IEEE Transactions on Image Processing, 16(3), 741-758.

EME measures the average local contrast in an image by dividing it into
non-overlapping blocks and computing the Weber-law ratio for each block:

    EME = (2 / (k₁ · k₂)) · Σ_{l,k} 20 · log(I_max(l,k) / (I_min(l,k) + ε))

where k₁ × k₂ is the number of blocks and I_max, I_min are the maximum and
minimum intensity values within each block.

Higher is better (more local contrast → better perceived enhancement).
No reference image is needed.
"""

import cv2
import numpy as np


def compute_eme(image: np.ndarray, block_size: int = 8) -> float:
    """Compute EME score for a single image.

    Args:
        image:      Input BGR image (uint8, H×W×3).
        block_size: Side length (in pixels) of each non-overlapping block.

    Returns:
        EME score. Higher is better.
    """
    # Work on greyscale / luminance
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    h, w = gray.shape
    # Crop to a multiple of block_size
    h_crop = (h // block_size) * block_size
    w_crop = (w // block_size) * block_size
    gray = gray[:h_crop, :w_crop]

    k1 = h_crop // block_size      # number of row-blocks
    k2 = w_crop // block_size      # number of col-blocks

    # Reshape into (k1, block_size, k2, block_size) → (k1, k2, block_size²)
    blocks = (gray.reshape(k1, block_size, k2, block_size)
                  .transpose(0, 2, 1, 3)
                  .reshape(k1, k2, -1))

    i_max = blocks.max(axis=2)
    i_min = blocks.min(axis=2)

    # Weber contrast ratio in log domain; ε avoids log(0)
    ratio = i_max / (i_min + 1e-7)
    eme = (2.0 / (k1 * k2)) * np.sum(20.0 * np.log(ratio + 1e-7))
    return float(eme)
