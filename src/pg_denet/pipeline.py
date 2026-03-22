"""HDR image processing pipeline orchestration.

Provides :func:`process_one_image` which runs the full per-image pipeline
(LLE → TM → YOLO → Metrics) and :func:`build_combined_avg` which aggregates
metrics across images / TM methods for chart generation.
"""

from __future__ import annotations

import gc
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from pg_denet.pre_processing import auto_expose, resize_max
from pg_denet.detection import detect, build_pseudo_gt, compute_perception_metrics
from pg_denet.metrics import compute_niqe, compute_eme, compute_brisque
from pg_denet.utils import timed, print_table


def process_one_image(
    path: Path,
    hdr_linear: np.ndarray,
    per_tm_metrics: dict[str, dict[str, dict[str, list]]],
    lle_methods: OrderedDict,
    tm_methods: OrderedDict,
    max_side: int = 1024,
    save_dir: Path | None = None,
) -> None:
    """Run the full pipeline on a single HDR image.

    Args:
        path:           Source file path (used for naming output files).
        hdr_linear:     Raw float32 linear BGR image.
        per_tm_metrics: Accumulator ``{tm: {lle: {"perc": [...], "iq": [...]}}}}``.
        lle_methods:    ``OrderedDict[name, fn]`` of LLE enhancement functions.
        tm_methods:     ``OrderedDict[name, fn]`` of Tone Mapping functions.
        max_side:       Resize longest side to this value.
        save_dir:       If given, save each LDR result image under
                        ``{save_dir}/{stem}/{tm_name}/{lle_name}.png``.
    """
    stem = path.stem
    print(f"\n{'=' * 64}")
    print(f"  {path.name}  (original {hdr_linear.shape[1]}×{hdr_linear.shape[0]})")
    print(f"{'=' * 64}")

    # ── 1. Pre-Processing: Linear Space ───────────────────────────────────
    hdr_linear = resize_max(hdr_linear, max_side)
    print(f"  Resized → {hdr_linear.shape[1]}×{hdr_linear.shape[0]}")
    hdr_exposed = auto_expose(hdr_linear, key=0.18)

    # ── 2. LLE Methods: enhance on float32 linear data ─────────────────
    print("\n[LLE Methods]")
    lle_results: OrderedDict[str, np.ndarray] = OrderedDict()
    for lle_name, lle_fn in lle_methods.items():
        lle_results[lle_name] = timed(lle_name, lle_fn, hdr_exposed)
    del hdr_exposed  # no longer needed after LLE

    # ── 3 & 4. TM → LDR → Metrics (per TM category) ─────────────────────
    for tm_name, tm_fn in tm_methods.items():
        print(f"\n{'─' * 64}")
        print(f"  Tone Mapping: {tm_name}")
        print(f"{'─' * 64}")

        tm_results: OrderedDict[str, np.ndarray] = OrderedDict()
        for lle_name, enhanced in lle_results.items():
            tm_results[lle_name] = timed(f"{lle_name}→{tm_name}", tm_fn, enhanced)

        # ── Save recovered LDR images (optional) ─────────────────────
        if save_dir is not None:
            for lle_name, img in tm_results.items():
                out_path = save_dir / path.stem / tm_name / f"{lle_name}.png"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_path), img)

        # ── YOLO Detection ────────────────────────────────────────────
        print("\n  [YOLO Detection]")
        detections: dict[str, dict] = {}
        for name, img in tm_results.items():
            detections[name] = timed(name, detect, img)

        # ── Perception Metrics ────────────────────────────────────────
        pseudo_gt = build_pseudo_gt(detections)
        perc_data: OrderedDict[str, dict[str, float]] = OrderedDict()
        perc_rows: list[list[str]] = []
        for name in tm_results:
            pm = compute_perception_metrics(detections[name], pseudo_gt)
            perc_data[name] = {
                "mAP@0.75": pm["mAP@0.75"],
                "mAP_small": pm["mAP_small"] if not np.isnan(pm["mAP_small"]) else 0.0,
                "Conf Mean": pm["Conf Mean"],
            }
            perc_rows.append([
                name,
                str(detections[name]["num_detections"]),
                f"{pm['mAP@0.75']:.4f}",
                f"{pm['mAP_small']:.4f}" if not np.isnan(pm["mAP_small"]) else "N/A",
                f"{pm['Conf Mean']:.4f}",
            ])
        print_table(
            f"Perception Metrics [{tm_name}]",
            ["LLE Method", "#Det", "mAP@0.75", "mAP_small", "Conf Mean"],
            perc_rows,
        )

        # ── Image Quality Metrics (No-Reference) ─────────────────────
        iq_data: OrderedDict[str, dict[str, float]] = OrderedDict()
        iq_rows: list[list[str]] = []
        for name, img in tm_results.items():
            niqe = compute_niqe(img)
            eme = compute_eme(img)
            brisque = compute_brisque(img)
            iq_data[name] = {"NIQE": niqe, "EME": eme, "BRISQUE": brisque}
            iq_rows.append([name, f"{niqe:.4f}", f"{eme:.4f}", f"{brisque:.4f}"])
        print_table(
            f"Image Quality Metrics [{tm_name}]",
            ["LLE Method", "NIQE ↓", "EME ↑", "BRISQUE ↓"],
            iq_rows,
        )

        # Accumulate metrics
        for lle_name in perc_data:
            per_tm_metrics[tm_name][lle_name]["perc"].append(perc_data[lle_name])
            per_tm_metrics[tm_name][lle_name]["iq"].append(iq_data[lle_name])

        del tm_results, detections
        gc.collect()

    del lle_results
    gc.collect()


def build_combined_avg(
    per_tm: dict[str, dict[str, dict[str, list]]],
    tm_names: list[str],
    lle_names: list[str],
) -> dict[str, dict[str, float]]:
    """Average all 6 metrics across specified TM methods and all images.

    Args:
        per_tm:    ``{tm: {lle: {"perc": [...], "iq": [...]}}}``.
        tm_names:  Which TM methods to include.
        lle_names: LLE method names (keeps ordering).

    Returns:
        ``{lle_name: {metric_name: avg_value}}``.
    """
    result: dict[str, dict[str, float]] = {}
    for lle_name in lle_names:
        merged: dict[str, list[float]] = {}
        for tm_name in tm_names:
            for key in ("perc", "iq"):
                for record in per_tm[tm_name][lle_name][key]:
                    for m, v in record.items():
                        merged.setdefault(m, []).append(v)
        result[lle_name] = {m: float(np.mean(vals)) for m, vals in merged.items()}
    return result
