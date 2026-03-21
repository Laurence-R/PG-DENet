import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
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


def save_chart(
    data: dict[str, dict[str, float]],
    title: str,
    filepath: Path,
    lower_is_better: set[str] | None = None,
) -> None:
    """Generate and save a grouped bar chart for metrics.

    Args:
        data:    {method_name: {metric_name: value}}.
        title:   Chart title.
        filepath: Output PNG path.
        lower_is_better: Set of metric names where lower = better (shown in orange).
    """
    if lower_is_better is None:
        lower_is_better = set()

    methods = list(data.keys())
    metrics = list(next(iter(data.values())).keys())
    n_methods = len(methods)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4.5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    colors_higher = plt.cm.Blues(np.linspace(0.45, 0.85, n_methods))
    colors_lower  = plt.cm.Oranges(np.linspace(0.45, 0.85, n_methods))

    for ax, metric in zip(axes, metrics):
        vals = [data[m].get(metric, 0.0) for m in methods]
        is_lower = metric in lower_is_better
        colors = colors_lower if is_lower else colors_higher
        bars = ax.bar(range(n_methods), vals, color=colors)
        ax.set_xticks(range(n_methods))
        ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=8)
        direction = "↓" if is_lower else "↑"
        ax.set_title(f"{metric} ({direction})", fontsize=11, fontweight="bold")
        ax.set_ylabel(metric)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(filepath), dpi=150)
    plt.close(fig)
