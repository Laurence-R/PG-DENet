"""Global Tone Mapping operators for HDR → LDR conversion (GPU-accelerated).

All operators expect **float32 BGR images in linear space** (auto-exposed)
and return **uint8 BGR images** suitable for display.

References:
    - Logarithmic: Drago et al. (2003), Adaptive logarithmic mapping.
    - Linear: Simple max-normalization.
    - Reinhard: Reinhard et al. (2002), Photographic tone reproduction.
"""

import math

import numpy as np
import torch

from pg_denet.gpu import to_gpu


def _apply_gamma_gpu(img: torch.Tensor, gamma: float = 2.2) -> np.ndarray:
    """Gamma-correct on GPU and return uint8 numpy array."""
    out = torch.clamp(img, 0.0, 1.0).pow(1.0 / gamma).mul(255.0)
    return out.to(torch.uint8).cpu().numpy()


# ── Logarithmic ──────────────────────────────────────────────────────────────

def tone_map_logarithmic(
    hdr_bgr: np.ndarray,
    mu: float = 100.0,
    gamma: float = 2.2,
) -> np.ndarray:
    """Logarithmic global tone mapping (per-channel, GPU).

    .. math:: C_{out} = \\frac{\\log(1 + \\mu \\cdot C / C_{\\max})}{\\log(1 + \\mu)}

    Args:
        hdr_bgr: Auto-exposed linear float32 BGR image.
        mu:      Compression parameter (larger → more compression).
        gamma:   Display gamma.

    Returns:
        uint8 BGR image.
    """
    img = torch.clamp(to_gpu(hdr_bgr), min=0.0)
    img_max = img.max()
    if img_max > 0:
        result = torch.log1p(mu * img / img_max) / math.log(1.0 + mu)
    else:
        result = torch.zeros_like(img)
    return _apply_gamma_gpu(result, gamma)


# ── Linear ────────────────────────────────────────────────────────────────────

def tone_map_linear(
    hdr_bgr: np.ndarray,
    gamma: float = 2.2,
) -> np.ndarray:
    """Linear global tone mapping (per-channel max normalization, GPU).

    .. math:: C_{out} = C / \\max(C_{all})

    Args:
        hdr_bgr: Auto-exposed linear float32 BGR image.
        gamma:   Display gamma.

    Returns:
        uint8 BGR image.
    """
    img = to_gpu(hdr_bgr)
    img_max = img.max()
    result = img / img_max if img_max > 0 else img.clone()
    return _apply_gamma_gpu(result, gamma)


# ── Reinhard ──────────────────────────────────────────────────────────────────

def tone_map_reinhard(
    hdr_bgr: np.ndarray,
    gamma: float = 2.2,
) -> np.ndarray:
    """Reinhard global tone mapping (per-channel, GPU).

    .. math:: C_{out} = \\frac{C}{1 + C}

    Args:
        hdr_bgr: Auto-exposed linear float32 BGR image.
        gamma:   Display gamma.

    Returns:
        uint8 BGR image.
    """
    img = torch.clamp(to_gpu(hdr_bgr), min=0.0)
    result = img / (1.0 + img)
    return _apply_gamma_gpu(result, gamma)
