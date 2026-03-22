import cv2
import numpy as np
import rawpy
from collections.abc import Iterator
from pathlib import Path


def load_images(directory: Path) -> list[tuple[Path, np.ndarray]]:
    """從目錄遞迴載入所有 PNG/JPG 圖片。

    Returns:
        List of (file_path, image) tuples.
    """
    files = list(directory.rglob("*.[pj][np]g"))
    return [(file, cv2.imread(str(file))) for file in files]


def hdr_loader(directory: Path) -> Iterator[tuple[Path, np.ndarray]]:
    """逐張載入目錄內所有 ARW (Sony RAW) 影像，轉為 Linear Space float32。

    使用 generator 逐張讀取，避免同時將所有大型 RAW 影像載入記憶體。

    Yields:
        (file_path, linear_BGR_float32) 每次一張。
    """
    files = sorted(directory.glob("*.ARW"))
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
        yield f, bgr

