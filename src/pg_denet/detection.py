"""Object detection with YOLO and perception-based evaluation metrics.

Provides:
    - ``detect``                  — run YOLO inference on a single image
    - ``build_pseudo_gt``         — aggregate detections from multiple methods via NMS
    - ``compute_perception_metrics`` — mAP\@0.75, mAP_small, Confidence Mean
"""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

# ── Model cache ───────────────────────────────────────────────────────────────

_model_cache: dict[str, YOLO] = {}


def _get_model(model_name: str) -> YOLO:
    if model_name not in _model_cache:
        _model_cache[model_name] = YOLO(model_name)
    return _model_cache[model_name]


# ── Detection ─────────────────────────────────────────────────────────────────

def detect(
    image: np.ndarray,
    model_name: str = "yolo11n.pt",
    conf: float = 0.25,
) -> dict:
    """Run YOLO object detection on a BGR uint8 image.

    Returns:
        dict with keys *boxes*, *num_detections*, *confidence_mean*, *annotated*.
    """
    model = _get_model(model_name)
    results = model(image, conf=conf, verbose=False)

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


# ── Pseudo ground-truth via NMS ───────────────────────────────────────────────

def build_pseudo_gt(
    all_detections: dict[str, dict],
    iou_threshold: float = 0.5,
) -> list[dict]:
    """Aggregate detections from every method and apply per-class NMS.

    The resulting set of boxes serves as *pseudo ground truth* so that
    ``compute_perception_metrics`` can derive mAP-like scores even when
    no human annotation is available.
    """
    all_boxes: list[list] = []
    all_scores: list[float] = []
    all_classes: list[int] = []

    for det in all_detections.values():
        for b in det["boxes"]:
            all_boxes.append(b["bbox"])
            all_scores.append(b["confidence"])
            all_classes.append(b["class_id"])

    if not all_boxes:
        return []

    unique_classes = set(all_classes)
    gt_boxes: list[dict] = []

    for cls in unique_classes:
        idxs = [i for i, c in enumerate(all_classes) if c == cls]
        cls_boxes = [all_boxes[i] for i in idxs]
        cls_scores = [all_scores[i] for i in idxs]

        order = sorted(range(len(cls_scores)), key=lambda i: -cls_scores[i])
        suppressed: set[int] = set()
        keep: list[int] = []

        for i in order:
            if i in suppressed:
                continue
            keep.append(i)
            for j in order:
                if j in suppressed or j == i:
                    continue
                if _iou(cls_boxes[i], cls_boxes[j]) > iou_threshold:
                    suppressed.add(j)

        for k in keep:
            bbox = cls_boxes[k]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            gt_boxes.append({"bbox": bbox, "class_id": cls, "area": area})

    return gt_boxes


# ── Average Precision ─────────────────────────────────────────────────────────

def _compute_ap(
    pred_boxes: list[dict],
    gt_list: list[dict],
    iou_threshold: float = 0.75,
) -> float:
    """Compute Average Precision at a single IoU threshold."""
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

    # all-point interpolation
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap


# ── Public: perception metrics ────────────────────────────────────────────────

SMALL_AREA_THRESHOLD = 32 * 32  # COCO definition


def compute_perception_metrics(
    method_det: dict,
    pseudo_gt: list[dict],
) -> dict[str, float]:
    """Compute perception metrics for one method's detections vs pseudo-GT.

    Returns:
        dict with keys ``mAP@0.75``, ``mAP_small``, ``Conf Mean``.
    """
    pred = [
        {"bbox": b["bbox"], "confidence": b["confidence"], "class_id": b["class_id"]}
        for b in method_det["boxes"]
    ]

    map_75 = _compute_ap(pred, pseudo_gt, iou_threshold=0.75)

    small_gt = [g for g in pseudo_gt if g["area"] < SMALL_AREA_THRESHOLD]
    map_small = _compute_ap(pred, small_gt, iou_threshold=0.5) if small_gt else float("nan")

    return {
        "mAP@0.75": map_75,
        "mAP_small": map_small,
        "Conf Mean": method_det["confidence_mean"],
    }
