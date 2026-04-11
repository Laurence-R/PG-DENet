"""Object detection with YOLO and perception-based evaluation metrics.

Provides:
    - ``detect``                — run YOLO inference on a single image
    - ``warmup_model``          — warm up YOLO model (CUDA kernels + TensorRT)
    - ``compute_metrics``       — mAP@50-95, Conf Mean, FPS, Inference Time
    - ``run_batch_detection``   — batch detection with metrics and annotated output
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ── Model cache ───────────────────────────────────────────────────────────────

_model_cache: dict[str, YOLO] = {}


def _get_model(model_name: str) -> YOLO:
    if model_name not in _model_cache:
        _model_cache[model_name] = YOLO(model_name)
    return _model_cache[model_name]


def warmup_model(model_name: str, imgsz: int = 640, runs: int = 3) -> None:
    """Warm up a YOLO model to stabilise CUDA kernels / TensorRT before timing."""
    model = _get_model(model_name)
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(runs):
        model(dummy, conf=0.25, imgsz=imgsz, verbose=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"  Warm-up done: {model_name} ({runs} runs)")


# ── Detection ─────────────────────────────────────────────────────────────────

def detect(
    image: np.ndarray,
    model_name: str,
    conf: float = 0.25,
) -> dict:
    """Run YOLO object detection on a BGR uint8 image.

    Uses ``torch.cuda.synchronize()`` around inference to ensure
    wall-clock timing is accurate.  Also captures YOLO's internal
    ``results[0].speed`` breakdown (preprocess / inference / postprocess)
    for a fair comparison between PyTorch and TensorRT backends.

    Returns:
        dict with keys *boxes*, *num_detections*, *confidence_mean*,
        *pre_ms*, *inference_ms*, *post_ms*, *inf_time_ms* (wall-clock),
        *fps* (inference-only), *annotated*.
    """
    model = _get_model(model_name)
    _cuda = torch.cuda.is_available()
    if _cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    results = model(image, conf=conf, imgsz=640, verbose=False)
    if _cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # YOLO internal speed breakdown (ms)
    speed = results[0].speed  # {'preprocess': ..., 'inference': ..., 'postprocess': ...}
    pre_ms = speed.get("preprocess", 0.0)
    inference_ms = speed.get("inference", 0.0)
    post_ms = speed.get("postprocess", 0.0)

    boxes: list[dict] = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy().tolist()
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            boxes.append(
                {
                    "class_id": cls_id,
                    "class_name": r.names[cls_id],
                    "confidence": confidence,
                    "bbox": bbox,
                    "area": area,
                }
            )

    confs = [b["confidence"] for b in boxes]
    return {
        "boxes": boxes,
        "num_detections": len(boxes),
        "confidence_mean": float(np.mean(confs)) if confs else 0.0,
        "pre_ms": pre_ms,
        "inference_ms": inference_ms,
        "post_ms": post_ms,
        "inf_time_ms": elapsed * 1000.0,  # wall-clock total
        "fps": 1000.0 / inference_ms if inference_ms > 0 else 0.0,
        "annotated": results[0].plot() if results else image.copy(),
    }


# ── IoU helper ────────────────────────────────────────────────────────────────

def _iou(box1: list, box2: list) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter + 1e-7)


# ── Average Precision ─────────────────────────────────────────────────────────

_IOU_THRESHOLDS = np.arange(0.50, 1.00, 0.05)


def _compute_ap(
    pred_boxes: list[dict],
    gt_list: list[dict],
    iou_threshold: float = 0.75,
) -> float:
    if not gt_list:
        return 1.0 if not pred_boxes else 0.0
    if not pred_boxes:
        return 0.0

    preds = sorted(pred_boxes, key=lambda x: -x["confidence"])
    gt_matched = [False] * len(gt_list)
    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))

    for i, pred in enumerate(preds):
        best_iou = 0.0
        best_gt = -1
        for j, gt in enumerate(gt_list):
            if pred["class_id"] != gt["class_id"]:
                continue
            val = _iou(pred["bbox"], gt["bbox"])
            if val > best_iou:
                best_iou = val
                best_gt = j

        if best_iou >= iou_threshold and best_gt >= 0 and not gt_matched[best_gt]:
            tp[i] = 1
            gt_matched[best_gt] = True
        else:
            fp[i] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / len(gt_list)
    precision = cum_tp / (cum_tp + cum_fp + 1e-7)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap


# ── Public: compute 5 perception metrics ──────────────────────────────────────

def compute_metrics(det_result: dict, gt_conf: float = 0.7) -> dict[str, float]:
    """Compute perception metrics from a single detection result.

    Pseudo ground-truth is built by filtering boxes with confidence >= *gt_conf*
    from the same detection result.

    Returns:
        dict with keys ``mAP@50-95``, ``Conf Mean``,
        ``Inference Time (ms)``, ``FPS``.
    """
    all_boxes = det_result["boxes"]

    # Pseudo-GT = high-confidence detections
    gt = [
        {"bbox": b["bbox"], "class_id": b["class_id"]}
        for b in all_boxes if b["confidence"] >= gt_conf
    ]

    pred = [
        {"bbox": b["bbox"], "confidence": b["confidence"], "class_id": b["class_id"]}
        for b in all_boxes
    ]

    # mAP@50-95
    map_5095 = float(np.mean([
        _compute_ap(pred, gt, iou_threshold=t) for t in _IOU_THRESHOLDS
    ]))

    return {
        "mAP@50-95": map_5095,
        "Conf Mean": det_result["confidence_mean"],
        "Inference Time (ms)": det_result["inference_ms"],
        "FPS": det_result["fps"],
    }


# ── Batch detection ───────────────────────────────────────────────────────────

def run_batch_detection(
    images: list[tuple[Path, np.ndarray]],
    model_name: str,
    det_dir: Path,
    preproc_ms: list[float],
    *,
    conf: float = 0.25,
    gt_conf: float = 0.7,
    label: str = "",
) -> list[dict[str, float]]:
    """Run YOLO detection on a batch of images, save annotated results.

    Args:
        images:      List of (path, uint8_image) tuples.
        model_name:  YOLO model filename.
        det_dir:     Directory to save annotated images.
        preproc_ms:  Per-image preprocessing time (ms), used for E2E calculation.
        conf:        Detection confidence threshold.
        gt_conf:     Pseudo-GT confidence threshold.
        label:       Print label prefix (e.g. "RAW", "ENH").

    Returns:
        List of per-image metric dicts (mAP@50-95, Conf Mean, Inference/E2E Time, FPS).
    """
    det_dir.mkdir(parents=True, exist_ok=True)
    total = len(images)
    metrics_list: list[dict[str, float]] = []

    for i, (path, img) in enumerate(images, 1):
        det = detect(img, model_name, conf=conf)
        m = compute_metrics(det, gt_conf=gt_conf)
        m["E2E Time (ms)"] = preproc_ms[i - 1] + m["Inference Time (ms)"]
        metrics_list.append(m)
        cv2.imwrite(str(det_dir / f"{path.stem}.png"), det["annotated"])
        print(
            f"  [{label}][{i:>2}/{total}] {path.name}  "
            f"#det={det['num_detections']:>3}  "
            f"mAP={m['mAP@50-95']:.4f}  "
            f"E2E={m['E2E Time (ms)']:.1f}ms"
        )

    print(f"  Saved annotated → {det_dir}/")
    return metrics_list
