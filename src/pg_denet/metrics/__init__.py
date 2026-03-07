"""Image quality metrics for low-light image enhancement evaluation.

Full-reference metrics (需要 ground truth):
    - PSNR   ↑  dB，越高越好
    - SSIM   ↑  [−1, 1]，越高越好
    - LPIPS  ↓  [0, 1]，越低越好（感知相似度）

No-reference metrics (不需要 ground truth):
    - NIQE    ↓  越低越好（越接近自然影像統計）
    - EME     ↑  越高越好（局部對比度）
    - BRISQUE ↓  [0, 100]，越低越好
"""

from pg_denet.metrics.psnr_ssim import compute_psnr, compute_ssim
from pg_denet.metrics.lpips import compute_lpips
from pg_denet.metrics.niqe import compute_niqe
from pg_denet.metrics.eme import compute_eme
from pg_denet.metrics.brisque import compute_brisque

__all__ = [
    "compute_psnr",
    "compute_ssim",
    "compute_lpips",
    "compute_niqe",
    "compute_eme",
    "compute_brisque",
]
