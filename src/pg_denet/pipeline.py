"""HDR image processing pipeline orchestration.

Provides:
    - :func:`make_accumulators` — create empty metric/timing accumulators
    - :func:`process_one_image` — full per-image pipeline (LLE → TM → YOLO → Metrics)
    - :func:`build_perc_avg`    — average perception metrics across TMs / images
    - :func:`build_scalar_avg`  — average a scalar timing field across TMs / images
"""

from __future__ import annotations

import gc
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from pg_denet.pre_processing import auto_expose, resize_max
from pg_denet.detection import detect, build_pseudo_gt, compute_perception_metrics
from pg_denet.utils import timed_ms, print_table


# ── Accumulator factory ─────────────────────────────────────────────────────

def make_accumulators(
    tm_names: list[str],
    lle_names: list[str],
) -> tuple[dict, dict[str, list[float]]]:
    """Create empty accumulator structures.

    Returns:
        ``(per_tm_data, lle_timings)``

        *per_tm_data[tm][lle]* contains lists for:
            perc, tm_ms, det_ms, det_fps, e2e_ms, e2e_fps

        *lle_timings[lle]* is a flat list of per-image LLE elapsed ms.
    """
    per_tm_data = {
        tm: {lle: {
            "perc": [],
            "tm_ms": [],
            "det_ms": [],
            "det_fps": [],
            "e2e_ms": [],
            "e2e_fps": [],
        } for lle in lle_names}
        for tm in tm_names
    }
    lle_timings: dict[str, list[float]] = {lle: [] for lle in lle_names}
    return per_tm_data, lle_timings


# ── Per-image pipeline ──────────────────────────────────────────────────────

def process_one_image(
    path: Path,
    hdr_linear: np.ndarray,
    per_tm_data: dict,
    lle_timings: dict[str, list[float]],
    lle_methods: OrderedDict,
    tm_methods: OrderedDict,
    max_side: int = 1024,
    save_dir: Path | None = None,
    model_name: str = "yolo11n.pt",
) -> None:
    """Run the full pipeline on a single HDR image and accumulate metrics."""
    print(f"\n{'=' * 64}")
    print(f"  {path.name}  (original {hdr_linear.shape[1]}×{hdr_linear.shape[0]})")
    print(f"{'=' * 64}")

    # ── 1. Pre-Processing ───────────────────────────────────────────────
    hdr_linear = resize_max(hdr_linear, max_side)
    print(f"  Resized → {hdr_linear.shape[1]}×{hdr_linear.shape[0]}")
    hdr_exposed = auto_expose(hdr_linear, key=0.18)

    # ── 2. LLE Methods ─────────────────────────────────────────────────
    print("\n[LLE Methods]")
    lle_results: OrderedDict[str, np.ndarray] = OrderedDict()
    lle_ms_map: dict[str, float] = {}
    for lle_name, lle_fn in lle_methods.items():
        result, ms = timed_ms(lle_name, lle_fn, hdr_exposed)
        lle_results[lle_name] = result
        lle_ms_map[lle_name] = ms
        lle_timings[lle_name].append(ms)
    del hdr_exposed

    # ── 3 & 4. TM → LDR → YOLO → Metrics ─────────────────────────────
    for tm_name, tm_fn in tm_methods.items():
        print(f"\n{'─' * 64}")
        print(f"  Tone Mapping: {tm_name}")
        print(f"{'─' * 64}")

        # Tone mapping
        tm_results: OrderedDict[str, np.ndarray] = OrderedDict()
        tm_ms_map: dict[str, float] = {}
        for lle_name, enhanced in lle_results.items():
            result, ms = timed_ms(f"{lle_name}→{tm_name}", tm_fn, enhanced)
            tm_results[lle_name] = result
            tm_ms_map[lle_name] = ms

        # Save recovered LDR images
        if save_dir is not None:
            for lle_name, img in tm_results.items():
                out_path = save_dir / path.stem / tm_name / f"{lle_name}.png"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_path), img)

        # YOLO Detection
        print("\n  [YOLO Detection]")
        detections: dict[str, dict] = {}
        for name, img in tm_results.items():
            det_result, _ = timed_ms(name, detect, img, model_name)
            detections[name] = det_result

        # Perception Metrics + accumulate timing
        pseudo_gt = build_pseudo_gt(detections)
        table_rows: list[list[str]] = []
        for lle_name in tm_results:
            pm = compute_perception_metrics(detections[lle_name], pseudo_gt)
            det_ms = detections[lle_name]["inf_time_ms"]
            det_fps = detections[lle_name]["fps"]
            lle_ms = lle_ms_map[lle_name]
            tm_ms = tm_ms_map[lle_name]
            e2e_ms = lle_ms + tm_ms + det_ms

            rec = per_tm_data[tm_name][lle_name]
            rec["perc"].append({
                "mAP@50-95": pm["mAP@50-95"],
                "mAP_small": pm["mAP_small"],
                "Conf Mean": pm["Conf Mean"],
            })
            rec["tm_ms"].append(tm_ms)
            rec["det_ms"].append(det_ms)
            rec["det_fps"].append(det_fps)
            rec["e2e_ms"].append(e2e_ms)
            rec["e2e_fps"].append(1000.0 / e2e_ms if e2e_ms > 0 else 0.0)

            table_rows.append([
                lle_name,
                str(detections[lle_name]["num_detections"]),
                f"{pm['mAP@50-95']:.4f}",
                f"{pm['mAP_small']:.4f}",
                f"{pm['Conf Mean']:.4f}",
                f"{lle_ms:.1f}+{tm_ms:.1f}+{det_ms:.1f}={e2e_ms:.1f}",
            ])
        print_table(
            f"Metrics [{tm_name}]",
            ["LLE", "#Det", "mAP@50-95", "mAP_small", "Conf Mean", "E2E(ms)"],
            table_rows,
        )

        del tm_results, detections
        gc.collect()

    del lle_results
    gc.collect()


# ── Aggregation helpers ────────────────────────────────────────────────────

def build_perc_avg(
    per_tm: dict,
    tm_names: list[str],
    lle_names: list[str],
) -> dict[str, dict[str, float]]:
    """Average perception metrics across TMs and images.

    Returns ``{lle: {"mAP@50-95": avg, "mAP_small": avg, "Conf Mean": avg}}``.
    """
    result: dict[str, dict[str, float]] = {}
    for lle in lle_names:
        merged: dict[str, list[float]] = {}
        for tm in tm_names:
            for record in per_tm[tm][lle]["perc"]:
                for m, v in record.items():
                    merged.setdefault(m, []).append(v)
        result[lle] = {m: float(np.mean(vals)) for m, vals in merged.items()}
    return result


def build_scalar_avg(
    per_tm: dict,
    tm_names: list[str],
    lle_names: list[str],
    key: str,
) -> dict[str, float]:
    """Average a scalar timing field across TMs and images.

    Returns ``{lle: avg_value}``.
    """
    result: dict[str, float] = {}
    for lle in lle_names:
        vals: list[float] = []
        for tm in tm_names:
            vals.extend(per_tm[tm][lle][key])
        result[lle] = float(np.mean(vals)) if vals else 0.0
    return result
