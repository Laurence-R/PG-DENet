"""BRISQUE — Blind/Referenceless Image Spatial Quality Evaluator (no-reference).

Reference:
    Mittal, A., Moorthy, A. K., & Bovik, A. C. (2012).
    No-Reference Image Quality Assessment in the Spatial Domain.
    IEEE Transactions on Image Processing, 21(12), 4695-4708.

BRISQUE extracts locally normalised luminance coefficients (MSCN) and their
pairwise products, fits Asymmetric Generalised Gaussian Distributions (AGGD)
to them, and uses an SVR to predict a quality score.

Lower is better (0 = pristine, 100 = heavily distorted).
No reference image is needed.
"""

import numpy as np
import piq


def compute_brisque(image: np.ndarray) -> float:
    """Compute BRISQUE score for a single BGR image.

    Args:
        image: BGR image (uint8, H×W×3).

    Returns:
        BRISQUE score in [0, 100]. Lower is better.
    """
    import torch
    import cv2

    # piq expects a float32 RGB tensor in [0, 1], shape (1, 3, H, W)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    return float(piq.brisque(tensor, data_range=1.0).item())
