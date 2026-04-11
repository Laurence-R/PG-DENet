"""PG-DENet — 1st Stage Pipeline

Pipeline:
    HDR Image (SID) → Pre-Processing → LLE → Tone Mapping → YOLO → Metrics

Each YOLO model produces a full set of charts under result/figurations/{model}/.
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
)
from pg_denet.pipeline import process_one_image, make_accumulators
from pg_denet.charts import generate_all_charts

# ── Configuration ────────────────────────────────────────────────────────────
HDR_DIR = Path("data/sid/short")
MAX_SIDE = 1024
SAVE_SAMPLES = 5       # 儲存前 N 張影像的恢復結果

MODELS = ["yolo11l.pt", "yolo26l.pt"]

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


def run_for_model(model_name: str, total: int) -> None:
    """Run full pipeline and generate charts for a single YOLO model."""
    label = model_name.replace(".pt", "")
    fig_dir = Path("result/figurations") / label
    lle_names = list(LLE_METHODS.keys())
    tm_names = list(TM_METHODS.keys())

    per_tm_data, lle_timings = make_accumulators(tm_names, lle_names)

    for i, (path, hdr_linear) in enumerate(hdr_loader(HDR_DIR), 1):
        print(f"\n[{i}/{total}]")
        save_dir = Path("result/samples") if i <= SAVE_SAMPLES else None
        process_one_image(
            path, hdr_linear, per_tm_data, lle_timings,
            lle_methods=LLE_METHODS,
            tm_methods=TM_METHODS,
            max_side=MAX_SIDE,
            save_dir=save_dir,
            model_name=model_name,
        )

    print(f"\n[Charts — {label}]")
    generate_all_charts(label, fig_dir, per_tm_data, lle_timings, lle_names, tm_names)


def main() -> None:
    total = sum(1 for _ in HDR_DIR.glob("*.ARW"))
    print(f"Found {total} HDR image(s) in {HDR_DIR}")

    for model_name in MODELS:
        label = model_name.replace(".pt", "")
        print(f"\n{'=' * 64}")
        print(f"  Model: {label}")
        print(f"{'=' * 64}")
        run_for_model(model_name, total)

    print("\n" + "=" * 64)
    print("  Pipeline complete — all results saved to result/")
    print("=" * 64)


if __name__ == "__main__":
    main()
