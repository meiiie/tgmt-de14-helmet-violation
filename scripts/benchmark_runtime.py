"""Benchmark the shared runtime on a provenance manifest; same entry point on CPU/CUDA."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from de14.core import decide, VIOL, crop
from de14.headdet import square
from de14.metrics import evaluate_frames
from de14.runtime import build_runtime, load_profile


def sha256(path):
    with open(path, "rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()


class VramSampler:
    """Optional sampled process VRAM includes ORT; Torch peaks alone do not."""
    def __init__(self, cuda):
        self.stop = threading.Event(); self.peak = None; self.thread = None
        self.note = "CUDA VRAM not applicable on CPU/Intel GPU; Intel shared-memory usage is not measured"
        if not cuda: return
        try:
            import pynvml
            pynvml.nvmlInit(); self.nvml = pynvml
            self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
            self.note = "Sampled process memory every50ms; includes ORT allocations when driver reports per-process usage"
            self.thread = threading.Thread(target=self._poll, daemon=True); self.thread.start()
        except Exception as e:
            self.note = f"Process VRAM unavailable ({type(e).__name__}); Torch peaks exclude ONNX Runtime"

    def _poll(self):
        while not self.stop.is_set():
            total = 0; known = False
            try:
                for handle in self.handles:
                    for p in self.nvml.nvmlDeviceGetComputeRunningProcesses(handle):
                        if p.pid == os.getpid() and isinstance(p.usedGpuMemory, int) and p.usedGpuMemory < 2**60:
                            total += p.usedGpuMemory; known = True
                if known: self.peak = max(self.peak or 0, total / 2**20)
            except Exception:
                self.note = "NVML could not read process memory; Torch peaks exclude ONNX Runtime"
            self.stop.wait(.05)

    def close(self):
        self.stop.set()
        if self.thread: self.thread.join(timeout=2); self.nvml.nvmlShutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="optimized_cpu")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/runtime_profiles.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/benchmark/manifest.json")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/benchmarks/optimized_cpu")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--regression-images", type=Path, nargs=2, metavar=("UNHELMETED", "FAMILY"))
    args = parser.parse_args()
    if args.repeats < 1: parser.error("--repeats must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    groups = defaultdict(list)
    for row in manifest["frames"]:
        path = ROOT / row["image"]
        if sha256(path) != row["sha256"]: raise ValueError(f"Input hash mismatch: {path}")
        image = cv2.imread(str(path))
        if image is None: raise ValueError(f"Unreadable image: {path}")
        width, height = manifest["source_resolution"]
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
        groups[row["clip"]].append((row["frame"], image))
    for rows in groups.values(): rows.sort(key=lambda r: r[0])
    if not groups: raise ValueError("Empty benchmark manifest")
    config = load_profile(args.profile, args.config)
    start = time.perf_counter(); bundle = build_runtime(config); load_seconds = time.perf_counter()-start
    cuda = bundle.device.startswith("cuda")
    def sync():
        if cuda: torch.cuda.synchronize(bundle.device)
    sync()
    warm = next(iter(groups.values()))[0]
    verifier = getattr(bundle.classifier, "verifier", None)
    cache_parity = {"tested_crops": 0, "passed": None, "max_margin_difference": None}
    if verifier is not None and getattr(bundle.detector, "hd", None) is not None:
        parity_image = cv2.imread(str(args.regression_images[0])) if args.regression_images else warm[1]
        if parity_image is None: raise ValueError("Cannot read CLIP parity image")
        crops = [crop(parity_image, square(b, margin=.15), 32) for _, b in bundle.detector.hd(parity_image)]
        crops = [c for c in crops if c is not None][:8]
        if crops:
            saved = verifier.cache_text
            try:
                verifier.cache_text = False; old = np.asarray(verifier(crops))
                verifier.cache_text = True; new = np.asarray(verifier(crops))
            finally:
                verifier.cache_text = saved
            passed = bool(np.allclose(old, new, atol=1e-4, rtol=1e-5))
            cache_parity = {"tested_crops": len(crops), "passed": passed,
                            "max_margin_difference": float(np.max(np.abs(old-new)))}
            if not passed: raise RuntimeError(f"CLIP cached-text parity failed: {cache_parity}")
    start = time.perf_counter(); bundle.pipeline(fps=10, n_min=1).run([warm]); sync()
    warm_seconds = time.perf_counter()-start
    if cuda: torch.cuda.reset_peak_memory_stats(bundle.device)
    vram = VramSampler(cuda)
    counters_before = dict(verifier.stats) if verifier else {}
    predictions = []; repeated_predictions = []; latencies = []; runtimes = []; det_s = cls_s = 0.; calls = 0
    try:
        for repeat in range(args.repeats):
            current = []
            for clip, frames in sorted(groups.items()):
                sync(); start = time.perf_counter()
                # Frame-level decision evaluation. Track-level voting requires a separate longer-video protocol.
                result = bundle.pipeline(fps=10, n_min=1).run(frames, write_video=False, evidence=False)
                sync(); elapsed = time.perf_counter()-start
                runtimes.append(elapsed); latencies.extend(result["frame_latency_ms"])
                det_s += result["t_det"]; cls_s += result["t_cls"]; calls += result["classifier_calls"]
                for frame, records in result["per_frame"]:
                    vehicles = []
                    for tid, box, _, heads in records:
                        labels = [decide([p], 1)[0] for _, p in heads]
                        decision = ("không đội mũ" if "không đội mũ" in labels else
                                    "có đội mũ" if labels and all(l == "có đội mũ" for l in labels) else "chưa xác định")
                        vehicles.append({"box": box, "decision": decision, "n_riders": len(heads),
                                         "heads": [{"box": h, "p_no_helmet": p, "label": label}
                                                   for (h, p), label in zip(heads, labels)]})
                    current.append({"clip": clip, "frame": frame, "vehicles": vehicles})
                print(args.profile, "repeat", repeat+1, clip, f"{len(frames)} frames {elapsed:.3f}s", flush=True)
            repeated_predictions.append(current)
            if repeat == 0: predictions = current
    finally:
        vram.close()
    torch_peak_allocated = torch.cuda.max_memory_allocated(bundle.device)/2**20 if cuda else None
    torch_peak_reserved = torch.cuda.max_memory_reserved(bundle.device)/2**20 if cuda else None
    metrics = evaluate_frames(manifest["frames"], predictions)
    counter_delta = {k: verifier.stats[k]-v for k, v in counters_before.items()} if verifier else {}
    n_frames = len(latencies)
    regression = {}
    if args.regression_images:
        for case, path, expected in zip(("unhelmeted", "family"), args.regression_images, ([2], [1, 2, 4])):
            image = cv2.imread(str(path))
            if image is None: raise ValueError(f"Cannot read regression image: {path}")
            folder = args.out / case
            pipeline = bundle.pipeline(fps=1, n_min=1)
            result = pipeline.run([(0, image)], out_dir=str(folder), video_name=case, tracking=False)
            cv2.imwrite(str(folder / "annotated.png"), pipeline.last_vis)
            counts = sorted(r["n_riders"] for r in result["tracks"].values())
            labels = [decide([p], 1)[0] for _, _, _, heads in result["per_frame"][0][1] for _, p in heads]
            regression[case] = {"riders_per_motorcycle": counts, "head_labels": labels,
                                "passed": counts == expected and (case != "unhelmeted" or labels == [VIOL, VIOL])}
    versions = {}
    for package in ("torch", "ultralytics", "transformers", "onnxruntime", "onnxruntime-gpu", "openvino"):
        try: versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: pass
    checkpoint_paths = {"yolo": config.yolo_weights, "probe": config.probe_path}
    if config.head_enabled: checkpoint_paths["head"] = config.head_path
    if config.accurate and config.verify_with_clip:
        checkpoint_paths["clip"] = str(Path(config.clip_path)/"model.safetensors")
    report = {"profile": args.profile, "runtime": bundle.metadata(), "versions": versions,
              "manifest_sha256": sha256(args.manifest), "config_file_sha256": sha256(args.config),
              "checkpoint_sha256": {k: sha256(ROOT/p) for k, p in checkpoint_paths.items()},
              "model_loading_seconds": load_seconds, "warmup_seconds": warm_seconds,
              "repeats": args.repeats, "frame_samples": n_frames,
              "latency_ms": {"p50": float(np.percentile(latencies, 50)), "p95": float(np.percentile(latencies, 95)),
                             "min": min(latencies), "max": max(latencies)},
              "fps_compute": n_frames / sum(runtimes), "compute_wall_seconds": sum(runtimes),
              "sec_detection": det_s, "sec_classification": cls_s,
              "classifier_calls": calls, "clip_forward_counts_after_warmup": counter_delta,
              "metrics_frame_vehicle": metrics,
              "clip_cache_parity": cache_parity,
              "repeat_metrics": [evaluate_frames(manifest["frames"], p) for p in repeated_predictions],
              "vram": {"torch_peak_allocated_mib": torch_peak_allocated,
                       "torch_peak_reserved_mib": torch_peak_reserved,
                       "process_peak_sampled_mib": vram.peak, "note": vram.note},
              "regression_images": regression,
              "limits": ["Small correlated official-test smoke subset, not representative mAP or full-video accuracy",
                         "Timing includes model/association/tracking, excludes input decode, model load and output export",
                         "GT may omit parked/background motorcycles; unmatched violation predictions counted as FP",
                         "VRAM/Torch peaks exclude GPU allocations outside Torch unless NVML process sampling is available"]}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out/"predictions.json").write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("profile", "latency_ms", "fps_compute", "metrics_frame_vehicle", "regression_images")}, ensure_ascii=True), flush=True)
    if regression and not all(r["passed"] for r in regression.values()): raise SystemExit(1)


if __name__ == "__main__":
    main()
