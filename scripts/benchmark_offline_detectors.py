"""Compare YOLO26s/YOLO11s on the reported images with the same offline accuracy stages."""
import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import cv2
import torch
import ultralytics

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(4)
import app
from de14.core import decide, VIOL
from de14.detectors import HeadEnhancedDetector, YoloDetector
from de14.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unhelmeted", required=True, type=Path)
    parser.add_argument("--family", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/offline_yolo26")
    args = parser.parse_args()
    if args.repeats < 1: parser.error("--repeats must be positive")
    images = {}
    for name, path in [("unhelmeted", args.unhelmeted), ("family", args.family)]:
        image = cv2.imread(str(path))
        if image is None: raise ValueError(f"Cannot read {path}")
        images[name] = image
    start = time.perf_counter()
    detector26, classifier = app.get_models(app.BACKENDS[1], 1280, accurate=True)
    detectors = {"yolo26s": detector26,
                 "yolo11s": HeadEnhancedDetector(YoloDetector(weights="weights/yolo11s.pt", imgsz=1280, conf=.2), detector26.hd)}
    report = {"scope": "Two known regression images; no independent accuracy or video-FPS claim.",
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "ultralytics": ultralytics.__version__, "detector_device": detector26.det.device,
                              "dino_device": str(classifier.base.e.device), "clip_device": str(classifier.verifier.device),
                              "onnx_providers": detector26.hd.providers, "torch_threads": 4},
              "imgsz": 1280, "repeats": args.repeats, "model_loading_seconds": time.perf_counter()-start,
              "shared_stages": "DEIMv2 full head detector, DINOv3-S/probe, CLIP, association, ByteTrack",
              "checkpoint_sha256": {name: hashlib.sha256((ROOT/f"weights/{name}.pt").read_bytes()).hexdigest()
                                    for name in detectors},
              "models": {name: {"cases": {}} for name in detectors}}
    for name, detector in detectors.items():
        t = time.perf_counter()
        Pipeline(detector, classifier, fps=1, n_min=1).run([(0, images["unhelmeted"])])
        report["models"][name]["warmup_seconds"] = time.perf_counter()-t
        print(name, "warmup done", flush=True)
    for case, image in images.items():
        expected = [2] if case == "unhelmeted" else [1, 2, 4]
        runs = {name: [] for name in detectors}
        for repeat in range(args.repeats):
            # Alternate order to reduce a systematic warm-up/load-order advantage.
            order = list(detectors) if repeat % 2 == 0 else list(reversed(detectors))
            for name in order:
                detector = detectors[name]
                t = time.perf_counter(); raw = detector.det(image); raw_seconds = time.perf_counter()-t
                folder = args.out / name / case
                pipeline = Pipeline(detector, classifier, fps=1, n_min=1)
                result = pipeline.run([(0, image)], out_dir=str(folder), video_name=case)
                cv2.imwrite(str(folder / "annotated.png"), pipeline.last_vis)
                counts = sorted(t["n_riders"] for t in result["tracks"].values())
                labels = [decide([p], n_min=1)[0] for _, _, _, heads in result["per_frame"][0][1] for _, p in heads]
                passed = counts == expected and (case != "unhelmeted" or labels == [VIOL, VIOL])
                runs[name].append({"pipeline_seconds": result["sec"], "detection_with_deim_seconds": result["t_det"],
                                   "classification_seconds": result["t_cls"], "raw_yolo_seconds": raw_seconds,
                                   "raw_motorcycle_boxes": len(raw["motorcycle"]), "raw_person_boxes": len(raw["person"]),
                                   "riders_per_motorcycle": counts, "head_labels": labels, "passed": passed})
                print(case, name, repeat+1, json.dumps(runs[name][-1], ensure_ascii=True), flush=True)
        for name in detectors:
            report["models"][name]["cases"][case] = {
                "expected_riders_per_motorcycle": expected, "runs": runs[name],
                "pipeline_median_seconds": statistics.median(r["pipeline_seconds"] for r in runs[name]),
                "raw_yolo_median_seconds": statistics.median(r["raw_yolo_seconds"] for r in runs[name]),
                "all_passed": all(r["passed"] for r in runs[name])}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not all(c["all_passed"] for c in report["models"]["yolo26s"]["cases"].values()): raise SystemExit(1)


if __name__ == "__main__":
    main()
