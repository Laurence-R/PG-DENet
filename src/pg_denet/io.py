import cv2
import numpy as np
import rawpy
from collections.abc import Iterator
from pathlib import Path


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


def save_batch(
    images: list[tuple[Path, np.ndarray]],
    out_dir: Path,
) -> None:
    """將 (path, image) 列表批次儲存至指定目錄。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    for path, img in images:
        cv2.imwrite(str(out_dir / f"{path.stem}.png"), img)

