import cv2
import numpy as np
from pathlib import Path


def load_images(directory: Path) -> list[tuple[Path, np.ndarray]]:
    """從目錄遞迴載入所有 PNG/JPG 圖片。

    Returns:
        List of (file_path, image) tuples.
    """
    files = list(directory.rglob("*.[pj][np]g"))
    return [(file, cv2.imread(str(file))) for file in files]

def hdr_loader(directory: Path) -> list[tuple[Path, np.ndarray]]:
    """從目錄載入所有的 HDR 圖片，並轉成 Linear Space。
    
    Returns:
        List of (file_path, image(float32)) tuples.
    """
