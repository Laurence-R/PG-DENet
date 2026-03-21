import cv2
import numpy as np
import rawpy
from pathlib import Path


def load_images(directory: Path) -> list[tuple[Path, np.ndarray]]:
    """從目錄遞迴載入所有 PNG/JPG 圖片。

    Returns:
        List of (file_path, image) tuples.
    """
    files = list(directory.rglob("*.[pj][np]g"))
    return [(file, cv2.imread(str(file))) for file in files]

def hdr_loader(directory: Path) -> list[tuple[Path, np.ndarray]]:
    """從目錄載入所有 ARW (Sony RAW) 圖片，並轉成 Linear Space float32。

    使用 rawpy 進行去馬賽克，輸出為線性空間（gamma=1）、
    套用相機白平衡，不做自動亮度調整。

    Returns:
        List of (file_path, linear_BGR_float32) tuples.  值域依實際曝光而定。
    """
    files = sorted(directory.glob("*.ARW"))
    results: list[tuple[Path, np.ndarray]] = []
    for f in files:
        with rawpy.imread(str(f)) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                half_size=False,
                no_auto_bright=True,
                output_bps=16,
                gamma=(1, 1),          # linear space
            )
        linear = rgb.astype(np.float32) / 65535.0
        bgr = cv2.cvtColor(linear, cv2.COLOR_RGB2BGR)
        results.append((f, bgr))
    return results

