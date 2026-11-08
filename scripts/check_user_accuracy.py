"""Real-model acceptance checks on the two reported images; never train on them."""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(4)
import app
from de14.core import decide, VIOL
from de14.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unhelmeted", type=Path, required=True)
    parser.add_argument("--family", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/user_accuracy")
    args = parser.parse_args()
    detector, classifier = app.get_models(app.BACKENDS[1], 1280, accurate=True)
    report = {"scope": "Acceptance checks on two user images; not an independent accuracy benchmark.",
              "detector": detector.name,
              "trained_on_these_images": False, "cases": {}}
    for name, path, expected in [("unhelmeted", args.unhelmeted, [2]), ("family", args.family, [1, 2, 4])]:
        image = cv2.imread(str(path))
        if image is None: raise ValueError(f"Cannot read {path}")
        folder = args.out / name
        pipeline = Pipeline(detector, classifier, fps=1, n_min=1)
        start = time.perf_counter()
        result = pipeline.run([(0, image)], out_dir=str(folder), video_name=name, tracking=False)
        elapsed = time.perf_counter() - start
        cv2.imwrite(str(folder / "annotated.png"), pipeline.last_vis)
        counts = sorted(row["n_riders"] for row in result["tracks"].values())
        labels = [decide([p], n_min=1)[0] for _, _, _, heads in result["per_frame"][0][1] for _, p in heads]
        passed = counts == expected and (name != "unhelmeted" or labels == [VIOL, VIOL])
        report["cases"][name] = {"detected_riders_per_motorcycle": counts, "expected": expected,
                                  "head_labels": labels, "seconds_warm_models": elapsed, "passed": passed}
        print(name, json.dumps(report["cases"][name], ensure_ascii=True), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not all(case["passed"] for case in report["cases"].values()): raise SystemExit(1)


if __name__ == "__main__":
    main()
