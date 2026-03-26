"""evaluate.py — 低光圖像復原機器感知指標評估入口

使用方式:
    uv run evaluate.py

資料集結構（LOL-style）:
    data/
    └── eval15/
        └── low/    ← 低光輸入

支援的 LLE 方法: CLAHE, LIME, AGCWD, MSRCR
Tone Mapping: Logarithmic（預設，可修改 TM_FN）
支援的指標:
    機器感知: mAP@50-95 ↑  mAP_small ↑  Conf Mean ↑  FPS ↑  Inf Time (ms) ↓
"""

from collections import OrderedDict
from pathlib import Path

import numpy as np

from pg_denet import (
    apply_agcwd, apply_clahe, apply_lime, apply_msrcr, load_images,
    tone_map_logarithmic,
)
from pg_denet.detection import detect, build_pseudo_gt, compute_perception_metrics
from pg_denet.utils import print_table

# ── 路徑設定 ─────────────────────────────────────────────────────────────────
LOW_DIR = Path("data/eval15/low")

# ── Tone Mapping（LLE 輸出為 float32，需先 TM 才能送入 YOLO） ─────────────────
TM_FN = tone_map_logarithmic

# ── 要評估的方法 ──────────────────────────────────────────────────────────────
METHODS: OrderedDict[str, callable] = OrderedDict([
    ("CLAHE", apply_clahe),
    ("LIME",  apply_lime),
    ("AGCWD", apply_agcwd),
    ("MSRCR", apply_msrcr),
])

METRIC_KEYS = ["mAP@50-95", "mAP_small", "Conf Mean", "FPS", "Inf Time (ms)"]


def print_summary(results: dict[str, dict[str, list[float]]]) -> None:
    """印出所有方法的平均指標表格。"""
    methods = list(results.keys())
    col_w = 14
    header = f"{'Metric':<18}" + "".join(f"{m:>{col_w}}" for m in methods)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for metric in METRIC_KEYS:
        row = f"{metric:<18}"
        for method in methods:
            vals = results[method].get(metric, [])
            avg = float(np.mean(vals)) if vals else float("nan")
            row += f"{avg:>{col_w}.4f}"
        print(row)
    print("=" * len(header))


def main() -> None:
    images = load_images(LOW_DIR)
    print(f"找到 {len(images)} 張低光圖片")

    # results[method][metric] = list of per-image scores
    results: dict[str, dict[str, list[float]]] = {name: {} for name in METHODS}

    for i, (path, src) in enumerate(images):
        print(f"\n[{i+1}/{len(images)}] {path.name}")

        # LLE → TM → uint8
        enhanced = {name: TM_FN(fn(src)) for name, fn in METHODS.items()}

        # YOLO 偵測
        detections = {name: detect(img) for name, img in enhanced.items()}
        pseudo_gt = build_pseudo_gt(detections)

        # 計算感知指標
        row_data: list[list[str]] = []
        for method_name in METHODS:
            pm = compute_perception_metrics(detections[method_name], pseudo_gt)
            for metric in METRIC_KEYS:
                results[method_name].setdefault(metric, []).append(pm[metric])
            row_data.append([
                method_name,
                f"{pm['mAP@50-95']:.4f}",
                f"{pm['mAP_small']:.4f}",
                f"{pm['Conf Mean']:.4f}",
                f"{pm['FPS']:.1f}",
                f"{pm['Inf Time (ms)']:.1f}",
            ])
        print_table(
            f"Perception Metrics [{path.name}]",
            ["Method", "mAP@50-95", "mAP_small", "Conf Mean", "FPS", "Inf Time(ms)"],
            row_data,
        )

    print_summary(results)


if __name__ == "__main__":
    main()
