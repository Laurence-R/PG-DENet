import math
from pathlib import Path

import cv2
import numpy as np


def build_grid(
    images: dict[str, np.ndarray],
    max_cols: int = 5,
    label_height: int = 28,
) -> np.ndarray:
    """將所有圖片拼成一個網格並回傳，不顯示視窗。

    Args:
        images:       {標題: 圖片} 的字典。
        max_cols:     每列最多幾張。
        label_height: 每張圖片上方標題列的高度（px）。

    Returns:
        拼好的網格圖片（BGR uint8）。
    """
    items = list(images.items())
    n = len(items)
    cols = min(n, max_cols)
    rows = math.ceil(n / cols)

    ref_h, ref_w = items[0][1].shape[:2]

    grid_rows = []
    for row_idx in range(rows):
        row_cells = []
        for col_idx in range(cols):
            idx = row_idx * cols + col_idx
            if idx < n:
                title, img = items[idx]
                cell = cv2.resize(img, (ref_w, ref_h))
                if cell.ndim == 2:
                    cell = cv2.cvtColor(cell, cv2.COLOR_GRAY2BGR)
            else:
                cell = np.zeros((ref_h, ref_w, 3), dtype=np.uint8)
                title = ""

            label_bar = np.zeros((label_height, ref_w, 3), dtype=np.uint8)
            cv2.putText(label_bar, title,
                        (6, label_height - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
            row_cells.append(np.vstack([label_bar, cell]))

        grid_rows.append(np.hstack(row_cells))

    return np.vstack(grid_rows)


def show_images(
    images: dict[str, np.ndarray],
    window_title: str = "Results",
    max_cols: int = 5,
    label_height: int = 28,
    wait: bool = True,
) -> None:
    """將所有圖片拼成一個網格，顯示在同一個視窗中。

    Args:
        images:       {標題: 圖片} 的字典。
        window_title: 視窗名稱。
        max_cols:     每列最多幾張。
        label_height: 每張圖片上方標題列的高度（px）。
        wait:         若為 True，按任意鍵後才關閉視窗。
    """
    grid = build_grid(images, max_cols=max_cols, label_height=label_height)
    cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(window_title, 100, 100)
    cv2.imshow(window_title, grid)

    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def save_images(
    images: dict[str, np.ndarray],
    filename: str,
    output_dir: Path | str = Path("result"),
    max_cols: int = 5,
    label_height: int = 28,
) -> Path:
    """將所有圖片以網格方式排列（排列順序與 show_images 相同）並儲存至檔案。

    Args:
        images:     {標題: 圖片} 的字典。
        filename:   輸出檔名（例如 '111.png'）。
        output_dir: 輸出資料夾，預設為 'result/'。
        max_cols:   每列最多幾張。
        label_height: 標題列高度（px）。

    Returns:
        存檔的完整路徑。
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = build_grid(images, max_cols=max_cols, label_height=label_height)
    out_path = out_dir / filename
    cv2.imwrite(str(out_path), grid)
    return out_path
