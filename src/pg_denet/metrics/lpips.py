"""LPIPS — Learned Perceptual Image Patch Similarity.

Reference:
    Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. (2018).
    The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.
    IEEE CVPR.

LPIPS measures perceptual similarity using deep feature activations (AlexNet
by default). Unlike PSNR/SSIM, it better correlates with human perception.

Range [0, 1]: lower is better (0 = identical).
"""

import cv2
import numpy as np
import torch
import lpips as lpips_lib


# Module-level model cache — loaded once on first use
_lpips_model: lpips_lib.LPIPS | None = None


def _get_model(net: str = "alex") -> lpips_lib.LPIPS:
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = lpips_lib.LPIPS(net=net, verbose=False)
        _lpips_model.eval()
    return _lpips_model


def _to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert uint8 BGR (H×W×3) → normalised float32 tensor (1×3×H×W) in [-1, 1]."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)


def compute_lpips(
    enhanced: np.ndarray,
    reference: np.ndarray,
    net: str = "alex",
) -> float:
    """Compute LPIPS between an enhanced image and a clean reference.

    Args:
        enhanced:  Enhanced BGR image (uint8, H×W×3).
        reference: Ground-truth BGR image (uint8, H×W×3).
        net:       Backbone network: 'alex' (default), 'vgg', or 'squeeze'.

    Returns:
        LPIPS score in [0, 1]. Lower is better.
    """
    model = _get_model(net)
    with torch.no_grad():
        score = model(_to_tensor(enhanced), _to_tensor(reference))
    return float(score.item())
