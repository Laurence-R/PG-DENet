"""Pre-processing utilities for the HDR image pipeline (GPU-accelerated).

Handles exposure normalization. No conversion to LDR here — that is the
job of the Tone Mapping stage.
"""

import cv2
import numpy as np
import torch

from pg_denet.gpu import luminance_gpu, to_gpu, to_cpu


def linear_to_uint8(img: np.ndarray) -> np.ndarray:
    """Naively convert linear float32 [0,1+] to uint8 by clipping."""
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def auto_expose(linear_bgr: np.ndarray, key: float = 0.18) -> np.ndarray:
    """Reinhard-style auto-exposure: maps log-average luminance to *key* value.

    GPU-accelerated: the log-average and scaling run on CUDA.

    Args:
        linear_bgr: Linear-space float32 BGR image.
        key:        Target middle-gray value (default 0.18).

    Returns:
        Exposed float32 BGR image (values may exceed 1.0).
    """
    img_t = to_gpu(linear_bgr)
    L = luminance_gpu(img_t)
    L_avg = torch.exp(torch.mean(torch.log(L + 1e-7)))
    scale = key / (L_avg + 1e-7)
    return to_cpu(img_t * scale)


def resize_max(image: np.ndarray, max_side: int = 1024) -> np.ndarray:
    """Resize so the longest side equals *max_side*, preserving aspect ratio.

    Images already smaller than *max_side* are returned unchanged.
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image
    scale = max_side / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(image, (new_w, new_h), interpolation=interp)
