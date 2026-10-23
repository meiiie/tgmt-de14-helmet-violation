"""Tạo crop đầu có nhãn từ các clip HELMET split 'training' (data/helmet/train_clips.txt).
Rider -> GT xe (IoA>=0.6); chỉ giữ xe mà mọi rider cùng trạng thái. Head crop = heuristic từ bbox person YOLO11."""
import os, glob, sys, numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__))
from de14.core import head_box, crop, ioa
from de14.detectors import YoloDetector
from de14.helmet_gt import load_clip, rider_counts
det = YoloDetector(imgsz=640, conf=0.2)
X, Y, meta = [], [], []
for clip in open("data/helmet/train_clips.txt").read().split():
    fr, _ = load_clip(clip)
    for f in sorted(glob.glob(f"data/helmet/frames/{clip}/*.jpg")):
        fid = int(os.path.basename(f)[:-4]); img = cv2.imread(f)
        if img is None or fid not in fr: continue
        gts = [(b, 1 if rider_counts(l)[1] else 0) for _, b, l in fr[fid] if 0 in rider_counts(l)]
        if not gts: continue
        for p in det(img)["person"]:
            g = max(gts, key=lambda g: ioa(p, g[0]))
            if ioa(p, g[0]) < 0.6: continue
            c = crop(img, head_box(p), 6)
            if c is None: continue
            X.append(cv2.resize(c, (64, 64))); Y.append(g[1]); meta.append(f"{clip}/{fid}")
os.makedirs("data/crops", exist_ok=True)
np.savez_compressed("data/crops/train_heads.npz", X=np.array(X), Y=np.array(Y), meta=np.array(meta))
print("crops", len(Y), "no_helmet", int(np.sum(Y)), "clips", len({m.split('/')[0] for m in meta}))
