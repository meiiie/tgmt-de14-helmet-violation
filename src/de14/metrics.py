"""Frame vehicle metrics for HELMET smoke benchmarks (not head AP or track metrics)."""
from __future__ import annotations

import math

VIOLATION = "không đội mũ"
SAFE = "có đội mũ"
UNKNOWN = "chưa xác định"


def _index(records, name):
    indexed = {}
    for record in records:
        key = (record["clip"], record["frame"])
        if key in indexed:
            raise ValueError(f"Duplicate {name} frame: {key}")
        indexed[key] = record["vehicles"]
    return indexed


def _box(vehicle):
    box = tuple(float(value) for value in vehicle["box"])
    if (len(box) != 4 or not all(math.isfinite(value) for value in box)
            or box[2] <= box[0] or box[3] <= box[1]):
        raise ValueError(f"Invalid xyxy box: {box}")
    riders = vehicle["n_riders"]
    if isinstance(riders, bool) or not isinstance(riders, int) or riders < 0:
        raise ValueError("n_riders must be a nonnegative integer")
    return box


def _iou(a, b):
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union


def evaluate_frames(frames, predictions, iou_threshold=0.3):
    """Evaluate original-resolution xyxy vehicles, independently in each frame.

    Match all candidate pairs in descending IoU order, with stable index tie
    breaking. Each ground-truth and predicted vehicle can be used once. A
    matched unknown/safe prediction on positive GT counts as a violation FN;
    every unmatched predicted violation counts as an FP. Unknown rate uses
    all predicted vehicles as denominator. Missing frames have no predictions.
    Rider errors use matched vehicles only, irrespective of helmet decision.
    Ratios with zero denominator are None (JSON null), never invented zeros.
    Extra prediction frames and duplicate frame records raise ValueError.
    """
    if not math.isfinite(iou_threshold) or not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    truth = _index(frames, "ground-truth")
    predicted = _index(predictions, "prediction")
    if predicted.keys() - truth.keys():
        raise ValueError("Prediction frames are outside the benchmark manifest")
    counts = dict(frames=len(truth), gt_vehicles=0, predicted_vehicles=0,
                  matched_vehicles=0, gt_violations=0, predicted_violations=0,
                  violation_tp=0, violation_fp=0, violation_fn=0,
                  unknown_decisions=0, rider_absolute_error=0, rider_exact=0)
    for key, gt_vehicles in truth.items():
        pred_vehicles = predicted.get(key, [])
        gt_boxes = [_box(vehicle) for vehicle in gt_vehicles]
        pred_boxes = [_box(vehicle) for vehicle in pred_vehicles]
        if any(type(vehicle["violation"]) is not bool for vehicle in gt_vehicles):
            raise ValueError("Ground-truth violation must be boolean")
        if any(vehicle["decision"] not in {VIOLATION, SAFE, UNKNOWN} for vehicle in pred_vehicles):
            raise ValueError("Unsupported vehicle decision")
        counts["gt_vehicles"] += len(gt_vehicles)
        counts["predicted_vehicles"] += len(pred_vehicles)
        counts["gt_violations"] += sum(vehicle["violation"] for vehicle in gt_vehicles)
        counts["predicted_violations"] += sum(vehicle["decision"] == VIOLATION for vehicle in pred_vehicles)
        counts["unknown_decisions"] += sum(vehicle["decision"] == UNKNOWN for vehicle in pred_vehicles)
        candidates = []
        for gt_index, gt_box in enumerate(gt_boxes):
            for pred_index, pred_box in enumerate(pred_boxes):
                overlap = _iou(gt_box, pred_box)
                if overlap >= iou_threshold:
                    candidates.append((-overlap, gt_index, pred_index))
        matched_gt, matched_pred = set(), set()
        for _, gt_index, pred_index in sorted(candidates):
            if gt_index in matched_gt or pred_index in matched_pred:
                continue
            matched_gt.add(gt_index)
            matched_pred.add(pred_index)
            gt, pred = gt_vehicles[gt_index], pred_vehicles[pred_index]
            counts["matched_vehicles"] += 1
            counts["violation_tp"] += gt["violation"] and pred["decision"] == VIOLATION
            error = abs(gt["n_riders"] - pred["n_riders"])
            counts["rider_absolute_error"] += error
            counts["rider_exact"] += error == 0
    counts["violation_fp"] = counts["predicted_violations"] - counts["violation_tp"]
    counts["violation_fn"] = counts["gt_violations"] - counts["violation_tp"]

    def ratio(numerator, denominator):
        return counts[numerator] / counts[denominator] if counts[denominator] else None

    return {
        "iou_threshold": iou_threshold,
        "counts": counts,
        "vehicle_detection_precision": ratio("matched_vehicles", "predicted_vehicles"),
        "vehicle_detection_recall": ratio("matched_vehicles", "gt_vehicles"),
        "violation_precision": ratio("violation_tp", "predicted_violations"),
        "violation_recall": ratio("violation_tp", "gt_violations"),
        "unknown_rate": ratio("unknown_decisions", "predicted_vehicles"),
        "rider_count_mae": ratio("rider_absolute_error", "matched_vehicles"),
        "rider_count_exact_rate": ratio("rider_exact", "matched_vehicles"),
    }
