"""Tải mẫu HELMET training để tạo probe trên máy mới (không dùng clip test).

Annotation và split: Kaggle mirror hozngvan/helmet-detection.
Frame: HF mirror eiglesiasr/nohelmetnet, theo quy trình get_data.sh.
Trọng số DINOv3 cần quyền truy cập riêng trên Hugging Face.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import requests


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips", type=int, default=40)
    ap.add_argument("--frames-per-clip", type=int, default=8)
    args = ap.parse_args()
    if args.clips < 5 or args.frames_per_clip < 1:
        ap.error("Cần ít nhất 5 clip cho GroupKFold và ít nhất 1 frame/clip.")
    root = Path(__file__).resolve().parents[1]
    data = root / "data/helmet"
    data.mkdir(parents=True, exist_ok=True)
    kg = "https://www.kaggle.com/api/v1/datasets/download/hozngvan/helmet-detection/"
    hf = "https://huggingface.co/datasets/eiglesiasr/nohelmetnet/resolve/main/"
    split_response = requests.get(kg + "data_split.csv", timeout=45)
    split_response.raise_for_status()
    split_text = split_response.content.decode("utf-8-sig")
    split = list(csv.DictReader(io.StringIO(split_text)))
    assert split and {"video_id", "Set"}.issubset(split[0]), "Sai định dạng split"
    train = {r["video_id"].lower(): r["video_id"] for r in split if r["Set"] == "training"}
    response = requests.get("https://huggingface.co/api/datasets/eiglesiasr/nohelmetnet", timeout=45)
    response.raise_for_status()
    grouped = defaultdict(list)
    for f in response.json()["siblings"]:
        name = f["rfilename"]
        m = re.fullmatch(r"images/[^/]+/(.+)_(\d+)\.jpg", name)
        if m and m[1] in train:
            grouped[train[m[1]]].append((int(m[2]), name))
    candidates = sorted(grouped)
    rng = random.Random(14)
    selected = sorted(rng.sample(candidates, min(args.clips, len(candidates))))
    assert len(selected) >= 5, "Không đủ clip training trong mirror"
    tasks = []
    for clip in selected:
        frames = sorted(grouped[clip])
        k = min(args.frames_per_clip, len(frames))
        ids = [round(i * (len(frames)-1) / max(1, k-1)) for i in range(k)]
        for i in ids:
            frame_id, name = frames[i]
            tasks.append((hf + quote(name, safe="/"), data / "frames" / clip / f"{frame_id:03d}.jpg"))
        ann = f"annotation/annotation/{clip}.csv"
        tasks.append((kg + quote(ann, safe=""), data / "ann/annotation/annotation" / f"{clip}.csv"))

    def download(task):
        url, dest = task
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.is_file():
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            if dest.suffix == ".jpg":
                assert r.content.startswith(b"\xff\xd8"), f"Không phải JPEG: {dest.name}"
            else:
                assert "track_id,frame_id" in r.content[:150].decode("utf-8-sig"), f"Sai annotation: {dest.name}"
            dest.write_bytes(r.content)
        return {"file": str(dest.relative_to(root)), "url": url,
                "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()}

    print(f"Preparing {len(selected)} official training clips, {len(tasks)-len(selected)} frames", flush=True)
    records = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, rec in enumerate(pool.map(download, tasks), 1):
            records.append(rec)
            if i % 40 == 0: print(f"Downloaded / checked {i}/{len(tasks)} files", flush=True)
    (data / "data_split.csv").write_text(split_text, encoding="utf-8")
    (data / "train_clips.txt").write_text("\n".join(selected) + "\n", encoding="utf-8")
    manifest = {"dataset": "HELMET", "split_source": kg + "data_split.csv",
                "split_sha256": hashlib.sha256(split_response.content).hexdigest(),
                "selection_seed": 14, "clips": selected, "frames_per_clip": args.frames_per_clip,
                "all_clips_official_training": all(c.lower() in train for c in selected),
                "files": records}
    (data / "probe_preparation.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("READY: official training clips only. Next: python src/build_head_crops.py", flush=True)


if __name__ == "__main__":
    main()
