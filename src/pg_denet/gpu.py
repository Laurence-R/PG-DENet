"""GPU utilities — thin wrapper around PyTorch CUDA tensors.

All image-processing modules convert *numpy* → *GPU tensor* → *numpy* at their
own boundaries so the rest of the pipeline stays numpy-based.
"""

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_gpu(arr: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Send a numpy array to GPU."""
    return torch.from_numpy(np.ascontiguousarray(arr)).to(device=device, dtype=dtype)


def to_cpu(t: torch.Tensor) -> np.ndarray:
    """Bring a GPU tensor back to numpy (float32)."""
    return t.detach().cpu().numpy().astype(np.float32)


def luminance_gpu(bgr: torch.Tensor) -> torch.Tensor:
    """Rec. 709 luminance from a (H, W, 3) BGR tensor."""
    return 0.0722 * bgr[:, :, 0] + 0.7152 * bgr[:, :, 1] + 0.2126 * bgr[:, :, 2]


def gaussian_blur_fft(channel: torch.Tensor, sigma: float) -> torch.Tensor:
    """FFT-based 2-D Gaussian blur on GPU — efficient for any sigma."""
    H, W = channel.shape
    # Build spatial-domain Gaussian kernel centred at (0,0) with wrap-around
    gy = torch.arange(H, device=channel.device, dtype=torch.float32)
    gx = torch.arange(W, device=channel.device, dtype=torch.float32)
    gy = torch.min(gy, H - gy)  # mirror distances for wrap-around
    gx = torch.min(gx, W - gx)
    yy, xx = torch.meshgrid(gy, gx, indexing="ij")
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()

    # Frequency-domain multiplication
    f_ch = torch.fft.rfft2(channel)
    f_k = torch.fft.rfft2(kernel)
    return torch.fft.irfft2(f_ch * f_k, s=(H, W))
