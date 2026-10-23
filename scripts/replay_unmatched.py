"""Tái lập tracking của Core v2 (SAM 3.1 cache + ByteTrack) để tìm track dự đoán KHÔNG khớp GT (IoU<0.3) và xuất box cho kiểm tra tay.
Tracking không phụ thuộc classifier -> dùng YOLO-helmet chỉ để chạy nhanh; quyết định lấy từ tracks.csv của v2."""
import sys, os, json, csv, glob
sys.path.insert(0, "src")
os.environ.setdefault("HELMET_FRAMES", "data/helmet/frames")
from eval_helmet import frames_of, match_tracks
from de14.helmet_gt import load_clip
from de14.pipeline import Pipeline
from de14.detectors import CachedSam3Detector
from de14.classifiers import YoloHelmetCls
CACHE, V2, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
cls = YoloHelmetCls("weights/helmet_yolo11s_nnsohamnn.pt"); res = {}
for cj in sorted(glob.glob(f"{CACHE}/*.json")):
    clip = os.path.basename(cj)[:-5]
    if not os.path.isdir(f"data/helmet/frames/{clip}"): print("skip", clip); continue
    det = CachedSam3Detector(cj, conf=0.4); gen = list(frames_of(clip, 2, set(det.c)))
    def it():
        for fi, k, img in gen: det.key = k; yield fi, img
    r = Pipeline(det, cls, fps=5, n_min=3, min_head=8).run(it(), out_dir=f"{OUT}/{clip}", video_name=clip, write_video=False)
    gtf, _ = load_clip(clip, W=1920, H=1080); m = match_tracks(r["per_frame"], gtf)
    v2 = {row["track_id"]: row for row in csv.DictReader(open(f"{V2}/{clip}/{clip}_tracks.csv"))}
    boxes = {}
    for fi, rec in r["per_frame"]:
        for tid, tb, *_ in rec: boxes.setdefault(str(tid), []).append([fi, [float(x) for x in tb]])
    res[clip] = {tid: {"matched_gt": m.get(int(tid)) if int(tid) in m else m.get(tid), "v2": v2.get(tid), "n_obs_replay": r["tracks"][int(tid) if int(tid) in r["tracks"] else tid].get("n_obs") if isinstance(r["tracks"].get(int(tid), r["tracks"].get(tid)), dict) else None,
                     "boxes": boxes[tid]} for tid in boxes}
    print(clip, "tracks", len(boxes), "unmatched", sum(v["matched_gt"] is None for v in res[clip].values()), flush=True)
json.dump(res, open(f"{OUT}/replay.json", "w"), ensure_ascii=False)
