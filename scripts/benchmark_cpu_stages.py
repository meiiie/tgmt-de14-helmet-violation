"""Profile OpenVINO head-thread settings on the same video samples and labeled frames."""
import argparse
from collections import defaultdict
from contextlib import ExitStack
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from de14 import pipeline as pipeline_module
from de14.headdet import DeimHeadDetector
from de14.metrics import evaluate_frames
from de14.runtime import build_runtime, load_profile
from de14.video_io import json_default, probe_video


def video_samples(source, info, first, count, stride):
    capture = cv2.VideoCapture(str(source))
    frames = []
    timeline = {}
    decode = 0.0
    try:
        if not capture.isOpened() or not capture.set(cv2.CAP_PROP_POS_FRAMES, first):
            raise ValueError("Could not seek to benchmark start")
        for offset in range((count - 1) * stride + 1):
            start = time.perf_counter()
            ok, image = capture.read()
            decode += time.perf_counter() - start
            if not ok:
                raise ValueError("Video ended before the benchmark window")
            if offset % stride == 0:
                index = len(frames)
                frames.append((index, image))
                timeline[index] = {"source_index": first + offset, "timestamp_s": info["timestamps"][first + offset]}
    finally:
        capture.release()
    return frames, timeline, decode


def profile_window(runtime, frames, timeline, fps, track_filter):
    costs = defaultdict(float)
    detector_samples = defaultdict(list)
    detector = runtime.detector
    head = detector.hd
    yolo = detector.det.m
    encoder = runtime.classifier.base.e
    yolo_calls = []

    def timer(name, call):
        def measured(*args, **kwargs):
            start = time.perf_counter()
            try:
                return call(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                costs[name] += elapsed
                if name in {"yolo_total_s", "deim_total_s"}:
                    detector_samples[name].append(elapsed)
        return measured

    predict = yolo.predict

    def measured_predict(*args, **kwargs):
        results = predict(*args, **kwargs)
        yolo_calls.append(results[0].speed)
        return results

    pipeline = runtime.pipeline(fps=fps, n_min=3, track_filter=track_filter)
    for frame in frames[:2]:
        pipeline.run([frame], tracking=False, write_video=False, evidence=False)
    head_time = head.t
    with ExitStack() as stack:
        for obj, attr, name in [
            (detector, "det", "yolo_total_s"),
            (detector, "hd", "deim_total_s"),
            (encoder, "proc", "dino_preprocessing_s"),
            (encoder.model, "forward", "dino_forward_s"),
            (pipeline_module.ByteTrackWrapper, "update", "tracker_update_s"),
            (pipeline_module, "crop", "crop_s"),
        ]:
            stack.enter_context(patch.object(obj, attr, timer(name, getattr(obj, attr))))
        stack.enter_context(patch.object(yolo, "predict", measured_predict))
        start = time.perf_counter()
        result = pipeline.run(frames, frame_metadata=timeline, write_video=False, evidence=False)
        wall = time.perf_counter() - start

    start = time.perf_counter()
    for (index, image), (_, records) in zip(frames, result["per_frame"]):
        pipeline_module.draw_annotations(image.copy(), records, result["frame_decisions"][index],
            timeline[index]["timestamp_s"], result["tracks"],
            untracked=result["untracked_detections"].get(index, []))
    costs["draw_sampled_frames_s"] = time.perf_counter() - start
    costs["deim_inference_s"] = head.t - head_time
    costs["deim_pre_and_post_s"] = costs["deim_total_s"] - costs["deim_inference_s"]
    costs["classifier_total_s"] = result["t_cls"]
    costs["association_and_aggregation_s"] = wall - result["t_det"] - result["t_cls"] - costs["tracker_update_s"] - costs["crop_s"]
    return {
        "video_compute_s": wall, "video_samples": len(frames), "fps_compute": len(frames) / wall,
        "latency_p50_s": float(np.percentile(result["frame_latency_ms"], 50)) / 1000,
        "latency_p95_s": float(np.percentile(result["frame_latency_ms"], 95)) / 1000,
        "frame_latency_ms": result["frame_latency_ms"],
        "detector_seconds_per_sample": dict(detector_samples),
        "stages": dict(costs),
        "yolo_ms_per_sample": {key: float(np.mean([row[key] for row in yolo_calls])) for key in yolo_calls[0]},
        "tracks": result["tracks"], "per_frame": result["per_frame"],
        "untracked": result["untracked_detections"],
    }


def quality(runtime, manifest):
    predictions = []
    for row in manifest["frames"]:
        path = ROOT / row["image"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Benchmark frame hash mismatch: {path}")
        image = cv2.resize(cv2.imread(str(path)), tuple(manifest["source_resolution"]), interpolation=cv2.INTER_CUBIC)
        result = runtime.pipeline(fps=1, n_min=1).run(
            [(0, image)], tracking=False, write_video=False, evidence=False)
        vehicles = []
        for tid, box, _, heads in result["per_frame"][0][1]:
            track = result["tracks"][tid]
            vehicles.append({"box": box, "decision": track["decision"], "n_riders": track["n_riders"], "heads": heads})
        predictions.append({"clip": row["clip"], "frame": row["frame"], "vehicles": vehicles})
    metrics = evaluate_frames(manifest["frames"], predictions)
    p, r = metrics["violation_precision"], metrics["violation_recall"]
    metrics["violation_f1"] = 2 * p * r / (p + r) if p is not None and r is not None and p + r else None
    return {"metrics": metrics, "predictions": predictions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", nargs="+", type=int, default=[4, 6, 8])
    parser.add_argument("--start", type=int, default=117)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/benchmark/manifest.json")
    parser.add_argument("--skip-quality", action="store_true")
    args = parser.parse_args()
    if min(args.samples, args.stride, *args.threads) < 1 or args.start < 0:
        parser.error("Samples, stride and threads must be positive; start must be nonnegative.")
    start = time.perf_counter()
    info = probe_video(args.source)
    probe = time.perf_counter() - start
    frames, timeline, decode = video_samples(args.source, info, args.start, args.samples, args.stride)
    manifest = json.loads(args.manifest.read_text(encoding="utf8"))
    track_filter = json.loads((ROOT / "configs/track_filter.json").read_text())["track_filter"]
    config = load_profile("openvino_cpu", accurate=False, detect_heads=True, ort_threads=args.threads[0])
    start = time.perf_counter()
    runtime = build_runtime(config)
    load = time.perf_counter() - start
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"source": str(args.source), "frame_indices": [v["source_index"] for v in timeline.values()],
              "probe_s": probe, "window_decode_s": decode, "baseline_model_load_s": load, "rows": []}
    for index, threads in enumerate(args.threads):
        if index:
            del runtime.detector.hd
            gc.collect()
            runtime.detector.hd = DeimHeadDetector(
                ROOT / config.head_path, device="cpu", min_score=.3, threads=threads,
                backend="openvino", openvino_device="CPU",
            )
        row = profile_window(runtime, frames, timeline, info["fps"] / args.stride, track_filter)
        row.update(threads=threads, effective_threads=runtime.detector.hd.compiled.get_property("INFERENCE_NUM_THREADS"))
        print(threads, "threads:", round(row["video_compute_s"], 3), "s", flush=True)
        if not args.skip_quality:
            row["quality"] = quality(runtime, manifest)
        report["rows"].append(row)
        (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=json_default), encoding="utf8")


if __name__ == "__main__":
    main()
