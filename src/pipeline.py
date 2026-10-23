"""Đề 14 pipeline skeleton: rider detect -> ByteTrack -> head crop -> helmet classify -> per-track voting."""
import argparse, csv, time
from collections import defaultdict
import cv2
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

def load_helmet(repo="nnsohamnn/helmet-detection-yolo11", fn="yolov11s(80 epochs).pt"):
    return YOLO(hf_hub_download(repo, fn))

def overlap(p, m):
    ix = max(0, min(p[2], m[2]) - max(p[0], m[0])); iy = max(0, min(p[3], m[3]) - max(p[1], m[1]))
    return ix * iy / max(1e-6, (p[2]-p[0])*(p[3]-p[1]))

def decide(votes, n_min=3, hi=0.6, lo=0.4):
    if len(votes) < n_min: return "chưa xác định"
    r = sum(votes) / len(votes)
    return "không đội mũ" if r >= hi else ("có đội mũ" if r <= lo else "chưa xác định")

def run(video, out_csv, stride=2, conf=0.35, min_head=24):
    det, helm = YOLO("yolo11n.pt"), load_helmet()
    votes = defaultdict(list); cap = cv2.VideoCapture(video); i = n = 0; t0 = time.time()
    while True:
        ok, f = cap.read()
        if not ok: break
        i += 1
        if i % stride: continue
        n += 1
        r = det.track(f, persist=True, tracker="bytetrack.yaml", classes=[0, 3], verbose=False)[0]
        if r.boxes.id is None: continue
        b = r.boxes; motos = [x for x, c in zip(b.xyxy.tolist(), b.cls.tolist()) if c == 3]
        for xy, c, tid in zip(b.xyxy.tolist(), b.cls.tolist(), b.id.int().tolist()):
            if c != 0 or not any(overlap(xy, m) > 0.2 for m in motos): continue
            x1, y1, x2, y2 = map(int, xy); h = y2 - y1
            crop = f[max(0, y1):y1 + h // 3, max(0, x1):x2]
            if crop.shape[0] < min_head or crop.shape[1] < min_head: continue
            hr = helm(crop, conf=conf, verbose=False)[0]
            if len(hr.boxes) == 0: continue
            k = int(hr.boxes.conf.argmax()); name = hr.names[int(hr.boxes.cls[k])].lower()
            votes[tid].append(1 if ("without" in name or "no" in name) else 0)
    fps = n / max(1e-6, time.time() - t0)
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["track_id", "n_votes", "no_helmet_ratio", "decision"])
        for tid, v in votes.items():
            w.writerow([tid, len(v), round(sum(v)/len(v), 3) if v else "", decide(v)])
    print(f"processed {n} frames, FPS={fps:.2f}, tracks={len(votes)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("--out", default="outputs/tracks.csv")
    ap.add_argument("--stride", type=int, default=2); a = ap.parse_args(); run(a.video, a.out, a.stride)
