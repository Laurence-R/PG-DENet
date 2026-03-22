"""LIME: Low-light IMage Enhancement via Illumination Map Estimation (GPU-accelerated).

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

GPU acceleration:
    Instead of building a huge sparse matrix (N×N, N≈700 K) and calling
    ``scipy.sparse.linalg.spsolve``, we use a **matrix-free Conjugate Gradient
    (CG)** solver on CUDA.  Each CG iteration evaluates the operator
    A·x = x + α·div(W·∇x) via simple element-wise + shift operations, which
    map perfectly to GPU parallelism.
"""

import numpy as np
import torch

from pg_denet.gpu import device, to_gpu, to_cpu


# ---------------------------------------------------------------------------
# Internal helpers (GPU)
# ---------------------------------------------------------------------------

def _apply_A(
    t: torch.Tensor,
    W_h: torch.Tensor,
    W_v: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Compute A·t = t + α·(Dh^T Wh Dh + Dv^T Wv Dv)·t  (matrix-free)."""
    # Forward differences
    dh = torch.zeros_like(t)
    dh[:, :-1] = t[:, 1:] - t[:, :-1]
    dv = torch.zeros_like(t)
    dv[:-1, :] = t[1:, :] - t[:-1, :]

    # Weighted
    w_dh = W_h * dh
    w_dv = W_v * dv

    # Adjoint (backward divergence)
    adj = torch.zeros_like(t)
    adj -= w_dh
    adj[:, 1:] += w_dh[:, :-1]
    adj -= w_dv
    adj[1:, :] += w_dv[:-1, :]

    return t + alpha * adj


def _cg_solve(
    b: torch.Tensor,
    W_h: torch.Tensor,
    W_v: torch.Tensor,
    alpha: float,
    tol: float = 1e-5,
    max_iter: int = 300,
) -> torch.Tensor:
    """Conjugate Gradient solver for (I + α·L)·x = b on GPU."""
    x = b.clone()
    r = b - _apply_A(x, W_h, W_v, alpha)
    p = r.clone()
    rs_old = torch.dot(r.flatten(), r.flatten())

    for _ in range(max_iter):
        Ap = _apply_A(p, W_h, W_v, alpha)
        pAp = torch.dot(p.flatten(), Ap.flatten())
        alpha_cg = rs_old / (pAp + 1e-10)
        x = x + alpha_cg * p
        r = r - alpha_cg * Ap
        rs_new = torch.dot(r.flatten(), r.flatten())
        if rs_new.sqrt().item() < tol:
            break
        p = r + (rs_new / (rs_old + 1e-10)) * p
        rs_old = rs_new

    return x


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_lime(
    image: np.ndarray,
    alpha: float = 0.15,
    gamma: float = 0.6,
    epsilon: float = 1e-3,
) -> np.ndarray:
    """Enhance a float32 linear-space BGR image using the LIME algorithm (GPU).

    Args:
        image:   Input float32 BGR 影像（線性空間，值可 > 1.0）。
        alpha:   Regularisation strength for illumination smoothing.
        gamma:   Gamma value for illumination correction.
        epsilon: Small constant preventing division-by-zero in weight computation.

    Returns:
        Enhanced float32 BGR 影像（線性空間）。
    """
    S_t = to_gpu(image)  # (H, W, 3)
    S_max = S_t.max()
    if S_max.item() < 1e-7:
        return image.copy()

    S_norm = S_t / S_max

    # --- Step 1: Initial illumination map T̂ = max_c(S) -------------------
    T_hat = S_norm.max(dim=2).values  # (H, W)

    # --- Step 2: Gradient weights -----------------------------------------
    dh = torch.zeros_like(T_hat)
    dh[:, :-1] = T_hat[:, 1:] - T_hat[:, :-1]
    dv = torch.zeros_like(T_hat)
    dv[:-1, :] = T_hat[1:, :] - T_hat[:-1, :]
    W_h = 1.0 / (torch.abs(dh) + epsilon)
    W_v = 1.0 / (torch.abs(dv) + epsilon)

    # --- Step 3: Refine illumination via CG on GPU ------------------------
    T = _cg_solve(T_hat, W_h, W_v, alpha)
    T = torch.clamp(T, 0.0, 1.0)

    # --- Step 4: Gamma correction -----------------------------------------
    T_gamma = torch.pow(T, gamma).unsqueeze(2)  # (H, W, 1)

    # --- Step 5: Enhance each channel  R_c = S_c / T_γ -------------------
    enhanced = S_norm / (T_gamma + 1e-7)
    result = enhanced * S_max
    return to_cpu(result)
