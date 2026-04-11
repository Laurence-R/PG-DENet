"""PG-DENet — 1st Stage Pipeline

Compares two approaches:
  A) RAW baseline: rawpy linear output → clip & scale to uint8 → YOLO
  B) Enhanced:     rawpy linear → auto_expose → CLAHE → Log TM → YOLO

Only first 20 images from SID/short.
"""

from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from pg_denet.io import hdr_loader, save_batch
from pg_denet.pre_processing import auto_expose, resize_max, linear_to_uint8
from pg_denet.lle_methods.clahe import apply_clahe
from pg_denet.tone_mapping import tone_map_logarithmic
from pg_denet.detection import warmup_model, run_batch_detection
from pg_denet.visualization import save_chart

# ── Configuration ────────────────────────────────────────────────────────────
HDR_DIR = Path("data/sid/short")
MODEL = "yolo26l.engine"
MAX_IMAGES = 20
MAX_SIDE = 1024
GT_CONF = 0.7
DET_CONF = 0.25

METRIC_KEYS = ["mAP@50-95", "Conf Mean", "Inference Time (ms)", "E2E Time (ms)", "FPS"]


def main() -> None:
    total = min(MAX_IMAGES, sum(1 for _ in HDR_DIR.glob("*.ARW")))
    print(f"Model: {MODEL}  |  Images: {total}  |  GT conf >= {GT_CONF}")
    print("=" * 70)

    # ── Phase 0: Load & Resize (CPU only) ─────────────────────────────────
    print("\n[Phase 0] Loading RAW images and resizing")
    raw_images: list[tuple[Path, np.ndarray]] = []
    for i, (path, hdr) in enumerate(hdr_loader(HDR_DIR), 1):
        if i > MAX_IMAGES:
            break
        hdr = resize_max(hdr, MAX_SIDE)
        raw_images.append((path, hdr))
        print(f"  [{i:>2}/{total}] {path.name}  →  {hdr.shape[1]}×{hdr.shape[0]}")

    # ── Phase 1A: RAW baseline (no enhancement) ──────────────────────────
    print("\n[Phase 1A] RAW Baseline: linear → clip to uint8")
    raw_ldr: list[tuple[Path, np.ndarray]] = []
    raw_preproc_ms: list[float] = []
    for path, hdr in raw_images:
        t0 = perf_counter()
        ldr = linear_to_uint8(hdr)
        pp_ms = (perf_counter() - t0) * 1000
        raw_preproc_ms.append(pp_ms)
        raw_ldr.append((path, ldr))
        print(f"  [{i:>2}/{total}] {path.name}  ({pp_ms:.1f}ms)")

    # ── Phase 1B: Enhanced pipeline (GPU) ─────────────────────────────────
    print("[Phase 1B] Enhanced: auto_expose → CLAHE → Logarithmic TM")
    _ = auto_expose(raw_images[0][1], key=0.18)
    torch.cuda.synchronize()

    enh_ldr: list[tuple[Path, np.ndarray]] = []
    preproc_ms: list[float] = []

    for i, (path, hdr) in enumerate(raw_images, 1):
        torch.cuda.synchronize()
        t0 = perf_counter()
        exposed = auto_expose(hdr, key=0.18)
        enhanced = apply_clahe(exposed)
        ldr = tone_map_logarithmic(enhanced)
        torch.cuda.synchronize()
        pp_ms = (perf_counter() - t0) * 1000
        preproc_ms.append(pp_ms)
        enh_ldr.append((path, ldr))
        print(f"  [{i:>2}/{total}] {path.name}  ({pp_ms:.1f}ms)")

    del raw_images

    # Save images for both pipelines
    raw_dir = Path("result/samples/raw_baseline")
    enh_dir = Path("result/samples/enhanced")
    save_batch(raw_ldr, raw_dir)
    save_batch(enh_ldr, enh_dir)
    print(f"  Saved RAW baseline → {raw_dir}/")
    print(f"  Saved Enhanced     → {enh_dir}/")

    # Release PyTorch GPU cache before TensorRT
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # ── Phase 2: YOLO Detection ───────────────────────────────────────────
    print(f"\n[Phase 2] YOLO Detection: {MODEL}")
    warmup_model(MODEL)

    print("\n--- A) RAW Baseline ---")
    raw_metrics = run_batch_detection(
        raw_ldr, MODEL, Path("result/samples/raw_baseline_det"),
        raw_preproc_ms, conf=DET_CONF, gt_conf=GT_CONF, label="RAW",
    )

    print("\n--- B) Enhanced (CLAHE + Log TM) ---")
    enh_metrics = run_batch_detection(
        enh_ldr, MODEL, Path("result/samples/enhanced_det"),
        preproc_ms, conf=DET_CONF, gt_conf=GT_CONF, label="ENH",
    )

    # ── Summary ───────────────────────────────────────────────────────────
    raw_avg = {k: float(np.mean([m[k] for m in raw_metrics])) for k in METRIC_KEYS}
    enh_avg = {k: float(np.mean([m[k] for m in enh_metrics])) for k in METRIC_KEYS}

    print(f"\n{'=' * 70}")
    print(f"  Comparison  ({MODEL}, {total} images)")
    print(f"{'=' * 70}")
    print(f"  {'Metric':<25s} {'RAW':>10s} {'Enhanced':>10s} {'Δ':>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    for k in METRIC_KEYS:
        r, e = raw_avg[k], enh_avg[k]
        delta = e - r
        fmt = ".1f" if "ms" in k or k == "FPS" else ".4f"
        sign = "+" if delta > 0 else ""
        print(f"  {k:<25s} {r:>10{fmt}} {e:>10{fmt}} {sign}{delta:>9{fmt}}")
    print(f"{'=' * 70}")

    # ── Chart ─────────────────────────────────────────────────────────────
    chart_data = {
        "RAW Baseline": raw_avg,
        "CLAHE + Log TM": enh_avg,
    }
    fig_path = Path("result/figurations/1st_stage_result.png")
    save_chart(
        chart_data,
        f"RAW vs Enhanced — {MODEL}",
        fig_path,
        lower_is_better={"Inference Time (ms)", "E2E Time (ms)"},
    )
    print(f"\n  Chart saved: {fig_path}")


if __name__ == "__main__":
    main()
