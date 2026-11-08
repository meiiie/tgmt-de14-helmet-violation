"""Prepare a deterministic HELMET official-test smoke subset; never change training data.

Default: four test clips, four consecutive frames each, seed 20261105.
The local official split is authoritative when present (test label: 'test');
fresh checkouts download the public split only into the benchmark directory.
HF mirror images are 640 square; consumers resize them to 1920x1080 to
match original annotation coordinates. This is not representative accuracy data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from de14.helmet_gt import is_violation, rider_counts

KAGGLE = "https://www.kaggle.com/api/v1/datasets/download/hozngvan/helmet-detection/"
HF = "https://huggingface.co/datasets/eiglesiasr/nohelmetnet/resolve/main/"
HF_METADATA = "https://huggingface.co/api/datasets/eiglesiasr/nohelmetnet"


def select_frames(grouped, clips=4, frames_per_clip=4, seed=20261105):
    """Sample only clips with consecutive frame windows, independent of labels/results."""
    if clips < 1 or frames_per_clip < 1:
        raise ValueError("clips and frames_per_clip must be positive")
    windows = {}
    for clip, available in sorted(grouped.items()):
        frames = sorted(available)
        if len({number for number, _ in frames}) != len(frames):
            raise ValueError(f"Duplicate mirror frame number for {clip}")
        starts = [i for i in range(len(frames) - frames_per_clip + 1)
                  if frames[i + frames_per_clip - 1][0] - frames[i][0] == frames_per_clip - 1]
        if starts:
            windows[clip] = (frames, starts)
    if len(windows) < clips:
        raise ValueError(f"Need {clips} test clips with consecutive windows; mirror has {len(windows)}")
    rng = random.Random(seed)
    selected = []
    for clip in sorted(rng.sample(sorted(windows), clips)):
        frames, starts = windows[clip]
        start = rng.choice(starts)
        selected.append((clip, frames[start:start + frames_per_clip]))
    return selected


def vehicles_for_frames(annotation, frame_ids):
    """Read original xywh annotations as original-resolution xyxy frame vehicles."""
    result = {frame: [] for frame in frame_ids}
    seen = set()
    reader = csv.DictReader(io.StringIO(annotation))
    required = {"track_id", "frame_id", "x", "y", "w", "h", "label"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError("Invalid HELMET annotation CSV columns")
    for row in reader:
        frame = int(row["frame_id"])
        if frame not in result:
            continue
        key = frame, row["track_id"]
        if key in seen:
            raise ValueError(f"Duplicate track within annotation frame: {key}")
        seen.add(key)
        x, y, width, height = (float(row[name]) for name in ("x", "y", "w", "h"))
        if not all(math.isfinite(value) for value in (x, y, width, height)) or min(width, height) <= 0:
            raise ValueError(f"Invalid annotation box: {key}")
        helmeted, unhelmeted = rider_counts(row["label"])
        n_riders = helmeted + unhelmeted
        if n_riders < 1:
            raise ValueError(f"No rider label in annotation: {key}")
        result[frame].append({"track_id": row["track_id"],
                              "box": [x, y, x + width, y + height],
                              "violation": is_violation(row["label"]), "n_riders": n_riders})
    return result


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def _fetch(url):
    response = requests.get(url, timeout=(15, 90))
    response.raise_for_status()
    return response.content


def _split_rows(content):
    rows = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
    if not rows or not {"video_id", "Set"}.issubset(rows[0]):
        raise ValueError("Invalid authoritative split CSV")
    return rows


def resolve_split(root, split_csv=None):
    """Use an explicit/local split, otherwise cache the public split under benchmark."""
    if split_csv is not None:
        explicit = Path(split_csv).resolve()
        if not explicit.is_file():
            raise FileNotFoundError(f"Explicit --split-csv file missing: {explicit}")
        return explicit
    local = root / "data/helmet/data_split.csv"
    if local.is_file():
        return local
    cached = root / "data/benchmark/source/data_split.csv"
    if not cached.is_file():
        content = _fetch(KAGGLE + "data_split.csv")
        _split_rows(content)  # Do not persist an HTML/error response as a CSV cache.
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(content)
    return cached


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=int, default=4)
    parser.add_argument("--frames-per-clip", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--split-csv", type=Path,
                        help="Explicit authoritative split CSV; a missing supplied file is an error")
    parser.add_argument("--ipv4", action="store_true", help="Use IPv4 if local IPv6 routes stall")
    args = parser.parse_args()
    if args.ipv4:
        # requests exposes no address-family option; limit this CLI process only.
        import socket
        import urllib3.util.connection
        urllib3.util.connection.allowed_gai_family = lambda: socket.AF_INET
    root = Path(__file__).resolve().parents[1]
    output = root / "data/benchmark"
    output.mkdir(parents=True, exist_ok=True)
    split_path = resolve_split(root, args.split_csv)
    split_content = split_path.read_bytes()
    rows = _split_rows(split_content)
    split_labels = Counter(row["Set"] for row in rows)
    # Official mirror currently calls this split 'test'; accept 'testing' too.
    testing = {row["video_id"].lower(): row["video_id"] for row in rows
               if row["Set"] in {"test", "testing"}}
    if not testing:
        raise ValueError(f"No official test rows; actual labels: {dict(split_labels)}")
    if len({row["video_id"].lower() for row in rows}) != len(rows):
        raise ValueError("Duplicate video IDs in authoritative split")
    metadata_path = output / "hf_metadata.json"
    if not metadata_path.is_file():
        metadata_path.write_bytes(_fetch(HF_METADATA))
    metadata_content = metadata_path.read_bytes()
    metadata = json.loads(metadata_content)
    grouped = defaultdict(list)
    for sibling in metadata["siblings"]:
        path = sibling["rfilename"]
        match = re.fullmatch(r"images/[^/]+/(.+)_(\d+)\.jpg", path)
        if match and match[1].lower() in testing:
            grouped[testing[match[1].lower()]].append((int(match[2]), path))
    selected = select_frames(grouped, args.clips, args.frames_per_clip, args.seed)
    print(f"Selected {len(selected)} official test clips: {[clip for clip, _ in selected]}", flush=True)
    split_record_path = (split_path.relative_to(root).as_posix() if split_path.is_relative_to(root)
                         else split_path.as_posix())
    files = [{"file": split_record_path,
              "url": KAGGLE + "data_split.csv", "sha256": _sha(split_content),
              "reuse_existing_read_only": split_path != output / "source/data_split.csv"},
             {"file": metadata_path.relative_to(root).as_posix(),
              "url": HF_METADATA, "sha256": _sha(metadata_content)}]
    frame_records = []
    tasks = []
    for clip, frames in selected:
        annotation_url = KAGGLE + quote(f"annotation/annotation/{clip}.csv", safe="")
        existing = root / "data/helmet/ann/annotation/annotation" / f"{clip}.csv"
        annotation_path = existing if existing.is_file() else output / "annotations" / f"{clip}.csv"
        if not annotation_path.is_file():
            content = _fetch(annotation_url)
            # Validate before persisting any HTTP response as an annotation.
            vehicles_for_frames(content.decode("utf-8-sig"), [number for number, _ in frames])
            annotation_path.parent.mkdir(parents=True, exist_ok=True)
            annotation_path.write_bytes(content)
        content = annotation_path.read_bytes()
        gt = vehicles_for_frames(content.decode("utf-8-sig"), [number for number, _ in frames])
        files.append({"file": annotation_path.relative_to(root).as_posix(),
                      "url": annotation_url, "sha256": _sha(content),
                      "reuse_existing_read_only": annotation_path == existing})
        for number, mirror_path in frames:
            image_path = output / "frames" / clip / f"{number:03d}.jpg"
            image_url = HF + quote(mirror_path, safe="/")
            tasks.append((image_path, image_url))
            frame_records.append({"clip": clip, "frame": number,
                                  "image": image_path.relative_to(root).as_posix(),
                                  "vehicles": gt[number]})

    def download_image(task):
        image_path, url = task
        content = image_path.read_bytes() if image_path.is_file() else _fetch(url)
        with Image.open(io.BytesIO(content)) as image:
            if image.format != "JPEG" or image.size != (640, 640):
                raise ValueError(f"Unexpected mirror image format/resolution: {image_path}")
            image.verify()
        image_path.parent.mkdir(parents=True, exist_ok=True)
        if not image_path.is_file():
            image_path.write_bytes(content)
        return {"file": image_path.relative_to(root).as_posix(), "url": url,
                "sha256": _sha(content), "mirror_resolution": [640, 640]}

    with ThreadPoolExecutor(max_workers=4) as pool:
        image_files = list(pool.map(download_image, tasks))
    files.extend(image_files)
    hashes = {record["file"]: record["sha256"] for record in image_files}
    for frame in frame_records:
        frame["sha256"] = hashes[frame["image"]]
    gt_vehicles = [vehicle for frame in frame_records for vehicle in frame["vehicles"]]
    manifest = {
        "schema_version": 1, "dataset": "HELMET", "split": "testing",
        "benchmark_kind": "smoke", "source_resolution": [1920, 1080],
        "mirror_resolution": [640, 640], "selection_seed": args.seed,
        "mirror_revision": metadata.get("sha"),
        "mirror_available_test_clips": len(grouped),
        "clips": [clip for clip, _ in selected], "frames_per_clip": args.frames_per_clip,
        "split_labels_in_source": dict(split_labels), "official_test_labels": ["test", "testing"],
        "selection": "Seeded sample of sorted available official-test clips with consecutive frame windows; seeded window per selected clip; no prediction or GT-label selection.",
        "limitations": [
            "Small smoke subset from available mirrored clips; not representative accuracy.",
            "Consecutive frames within a clip are correlated; frame metrics are not independent track metrics.",
            "HF mirror contains 640x640 resized images; resize to 1920x1080 before inference to match annotation boxes. Original image detail cannot be recovered.",
            "Mirrored GT may be incomplete. Unmatched predicted violations count as FP against available annotations.",
            "Local split CSV is authoritative; source test label is normalized to manifest testing.",
            "Frame vehicle detection/classification metrics are not head AP or track-level metrics."
        ],
        "counts": {"frames": len(frame_records), "gt_vehicles": len(gt_vehicles),
                   "gt_violations": sum(vehicle["violation"] for vehicle in gt_vehicles)},
        "files": files, "frames": frame_records,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": manifest_path.relative_to(root).as_posix(),
                      "counts": manifest["counts"]}), flush=True)


if __name__ == "__main__":
    main()
