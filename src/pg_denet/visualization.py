import math

import cv2
import numpy as np


def show_images(images: dict[str, np.ndarray],
                window_title: str = "Results",
                max_cols: int = 4,
                label_height: int = 28,
                wait: bool = True) -> None:
    """將所有圖片拼成一個網格，顯示在同一個視窗中。

    Args:
        images:       {標題: 圖片} 的字典。
        window_title: 視窗名稱。
        max_cols:     每列最多幾張。
        label_height: 每張圖片上方標題列的高度（px）。
        wait:         若為 True，按任意鍵後才關閉視窗。
    """
    items = list(images.items())
    n = len(items)
    cols = min(n, max_cols)
    rows = math.ceil(n / cols)

    # 統一所有圖片的尺寸（以第一張為基準）
    ref_h, ref_w = items[0][1].shape[:2]
    cell_h = ref_h + label_height

    grid_rows = []
    for row_idx in range(rows):
        row_cells = []
        for col_idx in range(cols):
            idx = row_idx * cols + col_idx
            if idx < n:
                title, img = items[idx]
                # 統一大小
                cell = cv2.resize(img, (ref_w, ref_h))
                # 若為灰階轉 BGR
                if cell.ndim == 2:
                    cell = cv2.cvtColor(cell, cv2.COLOR_GRAY2BGR)
            else:
                # 空白填充
                cell = np.zeros((ref_h, ref_w, 3), dtype=np.uint8)
                title = ""

            # 加上標題列
            label_bar = np.zeros((label_height, ref_w, 3), dtype=np.uint8)
            cv2.putText(label_bar, title,
                        (6, label_height - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
            row_cells.append(np.vstack([label_bar, cell]))

        grid_rows.append(np.hstack(row_cells))

    grid = np.vstack(grid_rows)
    
    cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(window_title, 100, 100)
    cv2.imshow(window_title, grid)

    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
