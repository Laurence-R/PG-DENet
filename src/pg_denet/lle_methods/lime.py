"""LIME: Low-light IMage Enhancement via Illumination Map Estimation.

Reference:
    Guo, X., Li, Y., & Ling, H. (2017).
    LIME: Low-light image enhancement via illumination map estimation.
    IEEE Transactions on Image Processing, 26(2), 982-993.

Algorithm overview (Fig. 1 of the paper):
    1. Estimate initial illumination map T̂ = max_{R,G,B}(S)
    2. Refine T̂ by solving a structure-aware L1 TV regularisation:
         min_T  ‖T − T̂‖²  +  α·(‖W_h ⊙ ∂_h T‖₁ + ‖W_v ⊙ ∂_v T‖₁)
       where the spatial weights W discourage smoothing across strong edges:
         W_h = 1 / (|∂_h T̂| + ε),  W_v = 1 / (|∂_v T̂| + ε)
    3. Gamma-correct the refined map:  T_γ = T ^ γ
    4. Recover the enhanced image:     R_c = S_c / T_γ  (per channel)
"""

import cv2
import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _initial_illumination(image: np.ndarray) -> np.ndarray:
    """Step 1 – T̂(x) = max_{c∈{R,G,B}} S_c(x).  Returns a (H,W) float32 map."""
    return np.max(image, axis=2)


def _gradient_weights(T_hat: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute structure-aware weights from the gradients of T̂.

    W_h(x) = 1 / (|∂_h T̂(x)| + ε)
    W_v(x) = 1 / (|∂_v T̂(x)| + ε)

    Forward differences with zero-padding at boundaries.
    """
    dh = np.diff(T_hat, axis=1, append=T_hat[:, -1:])   # (H, W)
    dv = np.diff(T_hat, axis=0, append=T_hat[-1:, :])   # (H, W)
    W_h = 1.0 / (np.abs(dh) + epsilon)
    W_v = 1.0 / (np.abs(dv) + epsilon)
    return W_h.astype(np.float32), W_v.astype(np.float32)


def _build_diff_matrices(h: int, w: int):
    """Build sparse N×N forward-difference matrices D_h and D_v (N = h*w).

    D_h implements horizontal forward difference with Dirichlet BC at right edge.
    D_v implements vertical   forward difference with Dirichlet BC at bottom edge.
    """
    n = h * w

    # ---- horizontal: pixel k ↔ (i, j),  difference = T[i,j+1] - T[i,j] ----
    mask_h = np.ones(n, dtype=np.float32)
    mask_h[np.arange(w - 1, n, w)] = 0.0          # no right neighbour at col w-1
    Dh = diags([-mask_h, mask_h[:-1]], [0, 1], shape=(n, n), format='csr')

    # ---- vertical:   pixel k ↔ (i, j),  difference = T[i+1,j] - T[i,j] ----
    mask_v = np.ones(n, dtype=np.float32)
    mask_v[n - w:] = 0.0                           # no bottom neighbour at row h-1
    Dv = diags([-mask_v, mask_v[:-w]], [0, w], shape=(n, n), format='csr')

    return Dh, Dv


def _refine_illumination(T_hat: np.ndarray,
                         W_h: np.ndarray,
                         W_v: np.ndarray,
                         alpha: float) -> np.ndarray:
    """Step 2 – Solve the weighted TV regularisation as a sparse linear system.

    Approximating the L1 TV with L2 (fixed-weight linearisation) gives:

        (I  +  α · (D_h^T W_h D_h  +  D_v^T W_v D_v)) · t  =  t̂

    where W_h, W_v are diagonal matrices built from the spatial weight maps.
    """
    h, w = T_hat.shape
    n = h * w

    Dh, Dv = _build_diff_matrices(h, w)

    Wh_diag = diags(W_h.flatten(), 0, format='csr')
    Wv_diag = diags(W_v.flatten(), 0, format='csr')

    A = eye(n, format='csr') + alpha * (Dh.T @ Wh_diag @ Dh + Dv.T @ Wv_diag @ Dv)
    b = T_hat.flatten()

    t = spsolve(A, b)
    return np.clip(t, 0.0, 1.0).reshape(h, w).astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_lime(
    image: np.ndarray,
    alpha: float = 0.15,
    gamma: float = 0.6,
    epsilon: float = 1e-3,
) -> np.ndarray:
    """Enhance a low-light BGR image using the LIME algorithm.

    Args:
        image:   Input BGR image (uint8, H×W×3).
        alpha:   Regularisation strength for illumination smoothing.
                 Larger → smoother T, less texture leakage.  (paper default: 0.15)
        gamma:   Gamma value for illumination correction.
                 Smaller → brighter output.  (paper default: 0.6 or tuned per-image)
        epsilon: Small constant preventing division-by-zero in weight computation.

    Returns:
        Enhanced BGR image (uint8, H×W×3).
    """
    # --- Normalise to [0, 1] float32 ------------------------------------
    S = image.astype(np.float32) / 255.0

    # --- Step 1: Initial illumination map --------------------------------
    T_hat = _initial_illumination(S)                    # (H, W)

    # --- Step 2: Compute spatial weights ---------------------------------
    W_h, W_v = _gradient_weights(T_hat, epsilon)

    # --- Step 3: Refine illumination map ---------------------------------
    T = _refine_illumination(T_hat, W_h, W_v, alpha)   # (H, W)

    # --- Step 4: Gamma correction ----------------------------------------
    T_gamma = np.power(T, gamma)                        # (H, W)

    # --- Step 5: Enhance each channel  R_c = S_c / T_γ ------------------
    T_gamma_3ch = np.stack([T_gamma] * 3, axis=2)      # broadcast to (H, W, 3)
    enhanced = np.clip(S / (T_gamma_3ch + 1e-7), 0.0, 1.0)

    return (enhanced * 255.0).astype(np.uint8)
