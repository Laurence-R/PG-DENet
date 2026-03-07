"""evaluate.py — 低光圖像復原品質評估入口

使用方式:
    uv run evaluate.py

資料集結構（LOL-style）:
    data/
    └── eval15/
        ├── low/    ← 低光輸入
        └── high/   ← 對應的乾淨 ground truth

支援的 LLE 方法: CLAHE, LIME, AGCWD, MSRCR
支援的指標:
    Full-reference : PSNR ↑  SSIM ↑  LPIPS ↓
    No-reference   : NIQE ↓  EME ↑  BRISQUE ↓
"""

import time
from pathlib import Path

import cv2
import numpy as np

from pg_denet import (
    apply_agcwd, apply_clahe, apply_lime, apply_msrcr, load_images,
)
from pg_denet.metrics import (
    compute_brisque, compute_eme, compute_lpips,
    compute_niqe, compute_psnr, compute_ssim,
)

# ── 路徑設定 ─────────────────────────────────────────────────────────────────
LOW_DIR  = Path("data/eval15/low")
HIGH_DIR = Path("data/eval15/high")

# ── 要評估的方法 ──────────────────────────────────────────────────────────────
METHODS: dict[str, callable] = {
    "CLAHE": apply_clahe,
    "LIME":  apply_lime,
    "AGCWD": apply_agcwd,
    "MSRCR": apply_msrcr,
}

# ── 指標定義 ──────────────────────────────────────────────────────────────────
# full-reference: (fn, 方向, 是否需要 reference)
# no-reference:   (fn, 方向, False)
METRICS_FR = {          # Full-Reference
    "PSNR  ↑": compute_psnr,
    "SSIM  ↑": compute_ssim,
    "LPIPS ↓": compute_lpips,
}
METRICS_NR = {          # No-Reference
    "NIQE    ↓": compute_niqe,
    "EME     ↑": compute_eme,
    "BRISQUE ↓": compute_brisque,
}


def _load_pair(low_path: Path, high_dir: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """載入低光圖與對應的 ground truth（若存在）。"""
    src = cv2.imread(str(low_path))
    ref_path = high_dir / low_path.name
    ref = cv2.imread(str(ref_path)) if ref_path.exists() else None
    if ref is not None and src.shape != ref.shape:
        ref = cv2.resize(ref, (src.shape[1], src.shape[0]))
    return src, ref


def evaluate_one(
    src: np.ndarray,
    ref: np.ndarray | None,
    method_name: str,
    method_fn: callable,
) -> dict[str, float]:
    """對單張圖片執行一個方法並計算所有指標。"""
    t0 = time.perf_counter()
    enhanced = method_fn(src)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    scores: dict[str, float] = {"time_ms": elapsed_ms}

    # Full-reference 指標
    if ref is not None:
        for name, fn in METRICS_FR.items():
            scores[name] = fn(enhanced, ref)

    # No-reference 指標
    for name, fn in METRICS_NR.items():
        scores[name] = fn(enhanced)

    return scores


def print_summary(results: dict[str, dict[str, list[float]]]) -> None:
    """印出所有方法的平均指標表格。"""
    methods = list(results.keys())
    metric_names = list(next(iter(results.values())).keys())

    col_w = 12
    header = f"{'Metric':<16}" + "".join(f"{m:>{col_w}}" for m in methods)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for metric in metric_names:
        row = f"{metric:<16}"
        for method in methods:
            vals = results[method].get(metric, [])
            avg = np.mean(vals) if vals else float("nan")
            if metric == "time_ms":
                row += f"{avg:>{col_w}.1f}"
            else:
                row += f"{avg:>{col_w}.4f}"
        print(row)

    print("=" * len(header))


def main() -> None:
    low_images = load_images(LOW_DIR)
    print(f"找到 {len(low_images)} 張低光圖片")

    # results[method][metric] = list of per-image scores
    results: dict[str, dict[str, list[float]]] = {
        name: {} for name in METHODS
    }

    for i, (low_path, src) in enumerate(low_images):
        src, ref = _load_pair(low_path, HIGH_DIR)
        has_ref = ref is not None
        print(f"\n[{i+1}/{len(low_images)}] {low_path.name}"
              + ("" if has_ref else "  (no GT)"))

        for method_name, method_fn in METHODS.items():
            scores = evaluate_one(src, ref, method_name, method_fn)
            for metric, val in scores.items():
                results[method_name].setdefault(metric, []).append(val)
            fr_str = " ".join(
                f"{k.split()[0]}={v:.3f}"
                for k, v in scores.items()
                if k != "time_ms" and k in {**METRICS_FR, **METRICS_NR}
            )
            print(f"  {method_name:<6} {scores['time_ms']:6.1f}ms  {fr_str}")

    print_summary(results)


if __name__ == "__main__":
    main()
