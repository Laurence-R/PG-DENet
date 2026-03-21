"""PG-DENet — 1st Stage Pipeline (Ver. 2)

Pipeline (see flowchart):
    HDR Image (SID)
        → Pre-Processing (Convert to Linear Space)
        → LLE Methods (CLAHE, MSRCR, AGCWD, LIME)  — enhance in linear space
        → Tone Mapping (Logarithmic / Linear / Reinhard) → LDR
        → YOLO11 → Perception Metrics  (mAP@0.75, mAP_small, Conf Mean)
        → Image Quality Metrics          (NIQE, EME, BRISQUE)

結果按 TM 方法分類，每個 TM 資料夾內比較 4 種 LLE 方法。
"""

from collections import OrderedDict
from pathlib import Path

from pg_denet import (
    hdr_loader,
    apply_clahe,
    apply_lime,
    apply_agcwd,
    apply_msrcr,
    tone_map_logarithmic,
    tone_map_linear,
    tone_map_reinhard,
    save_chart,
)
from pg_denet.pipeline import process_one_image, build_combined_avg

# ── Configuration ────────────────────────────────────────────────────────────
HDR_DIR = Path("data/hdr")
MAX_SIDE = 1024

TM_METHODS: OrderedDict[str, callable] = OrderedDict([
    ("Logarithmic", tone_map_logarithmic),
    ("Linear",      tone_map_linear),
    ("Reinhard",    tone_map_reinhard),
])

LLE_METHODS: OrderedDict[str, callable] = OrderedDict([
    ("CLAHE", apply_clahe),
    ("MSRCR", apply_msrcr),
    ("AGCWD", apply_agcwd),
    ("LIME",  apply_lime),
])


def main() -> None:
    hdr_images = hdr_loader(HDR_DIR)
    print(f"Loaded {len(hdr_images)} HDR image(s) from {HDR_DIR}")

    # 按 TM 追蹤: {tm_name: {lle_name: {"perc": [...], "iq": [...]}}}
    per_tm_metrics: dict[str, dict[str, dict[str, list]]] = {
        tm: {lle: {"perc": [], "iq": []} for lle in LLE_METHODS}
        for tm in TM_METHODS
    }

    for path, hdr_linear in hdr_images:
        process_one_image(
            path, hdr_linear, per_tm_metrics,
            lle_methods=LLE_METHODS,
            tm_methods=TM_METHODS,
            max_side=MAX_SIDE,
        )

    # 圖表儲存至 result/figurations/
    fig_dir = Path("result/figurations")
    lower_ib = {"NIQE", "BRISQUE"}
    lle_names = list(LLE_METHODS.keys())

    for tm_name in TM_METHODS:
        avg_data = build_combined_avg(per_tm_metrics, [tm_name], lle_names)
        save_chart(
            avg_data,
            f"Avg 6-Metric — {tm_name} (all images)",
            fig_dir / f"{tm_name}_avg.png",
            lower_is_better=lower_ib,
        )
        print(f"  ✓ result/figurations/{tm_name}_avg.png")

    all_avg = build_combined_avg(per_tm_metrics, list(TM_METHODS.keys()), lle_names)
    save_chart(
        all_avg,
        "Avg 6-Metric — All TM × All Images",
        fig_dir / "avg_all.png",
        lower_is_better=lower_ib,
    )
    print("  ✓ result/figurations/avg_all.png")

    print("\n" + "=" * 64)
    print("  Pipeline complete — all results saved to result/")
    print("=" * 64)


if __name__ == "__main__":
    main()
