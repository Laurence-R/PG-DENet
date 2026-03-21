"""Pre-processing utilities for the HDR image pipeline.

Handles exposure normalization. No conversion to LDR here — that is the
job of the Tone Mapping stage.
"""

import cv2
import numpy as np


def auto_expose(linear_bgr: np.ndarray, key: float = 0.18) -> np.ndarray:
    """Reinhard-style auto-exposure: maps log-average luminance to *key* value.

    Args:
        linear_bgr: Linear-space float32 BGR image.
        key:        Target middle-gray value (default 0.18).

    Returns:
        Exposed float32 BGR image (values may exceed 1.0).
    """
    L = 0.0722 * linear_bgr[:, :, 0] + 0.7152 * linear_bgr[:, :, 1] + 0.2126 * linear_bgr[:, :, 2]
    L_avg = np.exp(np.mean(np.log(L + 1e-7)))
    scale = key / (L_avg + 1e-7)
    return (linear_bgr * scale).astype(np.float32)


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
