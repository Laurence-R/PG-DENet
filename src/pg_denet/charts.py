"""Chart generation for the PG-DENet pipeline.

Generates 5 categories of charts per model:
    1. Perception metrics  — mAP@50-95, mAP_small, Conf Mean  (per TM + all)
    2. LLE processing time — single chart
    3. TM processing time  — per TM + all
    4. YOLO inference time & FPS — per TM + all
    5. End-to-end time & FPS    — per TM + all
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pg_denet.visualization import save_chart
from pg_denet.pipeline import build_perc_avg, build_scalar_avg


def generate_all_charts(
    label: str,
    fig_dir: Path,
    per_tm_data: dict,
    lle_timings: dict[str, list[float]],
    lle_names: list[str],
    tm_names: list[str],
) -> None:
    """Generate all chart categories for one model under *fig_dir*."""
    _gen_perception(label, fig_dir / "perc", per_tm_data, lle_names, tm_names)
    _gen_lle_timing(label, fig_dir, lle_timings)
    _gen_tm_timing(label, fig_dir / "timing_tm", per_tm_data, lle_names, tm_names)
    _gen_yolo_timing(label, fig_dir / "timing_yolo", per_tm_data, lle_names, tm_names)
    _gen_e2e_timing(label, fig_dir / "timing_e2e", per_tm_data, lle_names, tm_names)


# ── 1. Perception metrics ──────────────────────────────────────────────────

def _gen_perception(label, out_dir, per_tm_data, lle_names, tm_names):
    for tm in tm_names:
        data = build_perc_avg(per_tm_data, [tm], lle_names)
        out = out_dir / f"{tm}.png"
        save_chart(data, f"{label} — Perception — {tm}", out)
        print(f"  ✓ {out}")

    data = build_perc_avg(per_tm_data, tm_names, lle_names)
    out = out_dir / "all.png"
    save_chart(data, f"{label} — Perception — All TM", out)
    print(f"  ✓ {out}")


# ── 2. LLE timing ──────────────────────────────────────────────────────────

def _gen_lle_timing(label, fig_dir, lle_timings):
    data = {
        lle: {"LLE Time (ms)": float(np.mean(ts))}
        for lle, ts in lle_timings.items()
    }
    out = fig_dir / "timing_lle.png"
    save_chart(data, f"{label} — LLE Processing Time", out,
               lower_is_better={"LLE Time (ms)"})
    print(f"  ✓ {out}")


# ── 3. TM timing ───────────────────────────────────────────────────────────

def _gen_tm_timing(label, out_dir, per_tm_data, lle_names, tm_names):
    lib = {"TM Time (ms)"}
    for tm in tm_names:
        avgs = build_scalar_avg(per_tm_data, [tm], lle_names, "tm_ms")
        data = {lle: {"TM Time (ms)": v} for lle, v in avgs.items()}
        out = out_dir / f"{tm}.png"
        save_chart(data, f"{label} — TM Time — {tm}", out, lower_is_better=lib)
        print(f"  ✓ {out}")

    avgs = build_scalar_avg(per_tm_data, tm_names, lle_names, "tm_ms")
    data = {lle: {"TM Time (ms)": v} for lle, v in avgs.items()}
    out = out_dir / "all.png"
    save_chart(data, f"{label} — TM Time — All TM", out, lower_is_better=lib)
    print(f"  ✓ {out}")


# ── 4. YOLO inference timing ───────────────────────────────────────────────

def _gen_yolo_timing(label, out_dir, per_tm_data, lle_names, tm_names):
    lib = {"Inf Time (ms)"}
    for tm in tm_names:
        ms = build_scalar_avg(per_tm_data, [tm], lle_names, "det_ms")
        fps = build_scalar_avg(per_tm_data, [tm], lle_names, "det_fps")
        data = {lle: {"Inf Time (ms)": ms[lle], "FPS": fps[lle]} for lle in lle_names}
        out = out_dir / f"{tm}.png"
        save_chart(data, f"{label} — YOLO Timing — {tm}", out, lower_is_better=lib)
        print(f"  ✓ {out}")

    ms = build_scalar_avg(per_tm_data, tm_names, lle_names, "det_ms")
    fps = build_scalar_avg(per_tm_data, tm_names, lle_names, "det_fps")
    data = {lle: {"Inf Time (ms)": ms[lle], "FPS": fps[lle]} for lle in lle_names}
    out = out_dir / "all.png"
    save_chart(data, f"{label} — YOLO Timing — All TM", out, lower_is_better=lib)
    print(f"  ✓ {out}")


# ── 5. End-to-end timing ──────────────────────────────────────────────────

def _gen_e2e_timing(label, out_dir, per_tm_data, lle_names, tm_names):
    lib = {"E2E Time (ms)"}
    for tm in tm_names:
        ms = build_scalar_avg(per_tm_data, [tm], lle_names, "e2e_ms")
        fps = build_scalar_avg(per_tm_data, [tm], lle_names, "e2e_fps")
        data = {lle: {"E2E Time (ms)": ms[lle], "E2E FPS": fps[lle]} for lle in lle_names}
        out = out_dir / f"{tm}.png"
        save_chart(data, f"{label} — E2E Timing — {tm}", out, lower_is_better=lib)
        print(f"  ✓ {out}")

    ms = build_scalar_avg(per_tm_data, tm_names, lle_names, "e2e_ms")
    fps = build_scalar_avg(per_tm_data, tm_names, lle_names, "e2e_fps")
    data = {lle: {"E2E Time (ms)": ms[lle], "E2E FPS": fps[lle]} for lle in lle_names}
    out = out_dir / "all.png"
    save_chart(data, f"{label} — E2E Timing — All TM", out, lower_is_better=lib)
    print(f"  ✓ {out}")
