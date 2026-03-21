"""Global Tone Mapping operators for HDR → LDR conversion.

All operators expect **float32 BGR images in linear space** (auto-exposed)
and return **uint8 BGR images** suitable for display.

References:
    - Logarithmic: Drago et al. (2003), Adaptive logarithmic mapping.
    - Linear: Simple max-normalization.
    - Reinhard: Reinhard et al. (2002), Photographic tone reproduction.
"""

import numpy as np


def _luminance(bgr: np.ndarray) -> np.ndarray:
    """Rec. 709 luminance from BGR."""
    return 0.0722 * bgr[:, :, 0] + 0.7152 * bgr[:, :, 1] + 0.2126 * bgr[:, :, 2]


def _apply_gamma(img: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """Gamma-correct and convert to uint8."""
    return (np.clip(img, 0.0, 1.0) ** (1.0 / gamma) * 255).astype(np.uint8)


# ── Logarithmic ──────────────────────────────────────────────────────────────

def tone_map_logarithmic(
    hdr_bgr: np.ndarray,
    mu: float = 100.0,
    gamma: float = 2.2,
) -> np.ndarray:
    """Logarithmic global tone mapping.

    .. math:: L_{out} = \\frac{\\log(1 + \\mu \\, L)}{\\log(1 + \\mu)}

    Args:
        hdr_bgr: Auto-exposed linear float32 BGR image.
        mu:      Compression parameter (larger → more compression).
        gamma:   Display gamma.

    Returns:
        uint8 BGR image.
    """
    L = _luminance(hdr_bgr)
    L_mapped = np.log(1.0 + mu * L) / np.log(1.0 + mu)
    scale = L_mapped / (L + 1e-7)
    result = hdr_bgr * scale[:, :, np.newaxis]
    return _apply_gamma(result, gamma)


# ── Linear ────────────────────────────────────────────────────────────────────

def tone_map_linear(
    hdr_bgr: np.ndarray,
    gamma: float = 2.2,
) -> np.ndarray:
    """Linear global tone mapping (max-normalization).

    .. math:: L_{out} = L / \\max(L)

    Args:
        hdr_bgr: Auto-exposed linear float32 BGR image.
        gamma:   Display gamma.

    Returns:
        uint8 BGR image.
    """
    L = _luminance(hdr_bgr)
    L_max = L.max()
    result = hdr_bgr / L_max if L_max > 0 else hdr_bgr.copy()
    return _apply_gamma(result, gamma)


# ── Reinhard ──────────────────────────────────────────────────────────────────

def tone_map_reinhard(
    hdr_bgr: np.ndarray,
    gamma: float = 2.2,
) -> np.ndarray:
    """Reinhard global tone mapping.

    .. math:: L_{out} = \\frac{L}{1 + L}

    Args:
        hdr_bgr: Auto-exposed linear float32 BGR image.
        gamma:   Display gamma.

    Returns:
        uint8 BGR image.
    """
    L = _luminance(hdr_bgr)
    L_mapped = L / (1.0 + L)
    scale = L_mapped / (L + 1e-7)
    result = hdr_bgr * scale[:, :, np.newaxis]
    return _apply_gamma(result, gamma)
