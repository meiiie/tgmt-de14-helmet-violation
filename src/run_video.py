"""Run the same configured pipeline locally or in a Colab/Kaggle GPU notebook."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from de14.runtime import build_runtime, load_profile
from de14.video_io import run_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path)
    parser.add_argument("--profile", default="optimized_cpu")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--detector", choices=["yolo", "sam31"], default="yolo")
    parser.add_argument("--classifier", choices=["yolo", "dino"], default="dino")
    parser.add_argument("--no-accuracy", action="store_true", help="Skip CLIP cross-check; keep independent head localization")
    parser.add_argument("--estimated-heads", action="store_true", help="Skip DEIM and estimate head regions from person boxes")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=0, help="Inference limit only; 0 analyzes entire source")
    parser.add_argument("--min-observations", type=int, default=3)
    parser.add_argument("--f2", action="store_true", help="Apply existing F2 thresholds; audit all removed tracks")
    parser.add_argument("--output-mode", choices=["full", "sampled_preview"], default="full")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--out", type=Path, default=Path("outputs/demo"))
    args = parser.parse_args()
    if args.stride < 1: parser.error("--stride must be positive")
    overrides = {"imgsz": args.imgsz, "classifier_kind": args.classifier,
                 "detector_kind": "sam3.1" if args.detector == "sam31" else "yolo",
                 "accurate": not args.no_accuracy, "detect_heads": not args.estimated_heads}
    if args.detector == "sam31": overrides.update(device="cuda", head_backend="onnx")
    runtime = build_runtime(load_profile(args.profile, args.config, **overrides))
    videos = sorted(args.src.iterdir()) if args.src.is_dir() else [args.src]
    summary = {"runtime": runtime.metadata(), "videos": {}}
    for video in videos:
        filtering = json.loads(Path('configs/track_filter.json').read_text())['track_filter'] if args.f2 else None
        pipeline = runtime.pipeline(n_min=args.min_observations, track_filter=filtering)
        result = run_video(video, pipeline, args.out, video.stem, stride=args.stride,
                           max_seconds=args.max_seconds, max_frames=args.max_frames, output_mode=args.output_mode)
        counts = {}
        for track in result["tracks"].values():
            counts[track["decision"]] = counts.get(track["decision"], 0) + 1
        summary["videos"][video.name] = {"frames": result["frames"], "fps_compute": result["fps"],
                                       "classifier_calls": result["classifier_calls"], "tracks": counts,
                                       "media": result["media"], "f2": result["f2"], "video_path": result["video_path"],
                                       "total_seconds": result["total_seconds"]}
        print(video.name, json.dumps(summary["videos"][video.name], ensure_ascii=True), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
