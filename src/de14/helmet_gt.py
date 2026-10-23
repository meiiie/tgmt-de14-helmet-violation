"""Đọc nhãn HELMET (Lin & Siebert 2020, CC-BY 4.0; annotation từ Kaggle mirror hozngvan/helmet-detection)
và ảnh 640x640 (bản resize trên HF eiglesiasr/nohelmetnet). Toạ độ gốc 1920x1080 -> quy đổi."""
import csv, re, os
from collections import defaultdict, Counter

ANN = "data/helmet/ann/annotation/annotation"

def rider_counts(label):
    nh = label.count("NoHelmet"); tot = len(re.findall(r"(D|P\d)", label)); return tot - nh, nh

def load_clip(clip, W=640, H=640, W0=1920, H0=1080):
    """-> frames: {frame_id: [(track_id, xyxy, label)]}, tracks: {track_id: majority label}"""
    fr = defaultdict(list); lab = defaultdict(Counter)
    for r in csv.DictReader(open(os.path.join(ANN, f"{clip}.csv"))):
        f = int(r["frame_id"]); x, y, w, h = (float(r[k]) for k in "xywh")
        sx, sy = W / W0, H / H0
        fr[f].append((r["track_id"], [x*sx, y*sy, (x+w)*sx, (y+h)*sy], r["label"]))
        lab[r["track_id"]][r["label"]] += 1
    return fr, {t: c.most_common(1)[0][0] for t, c in lab.items()}

def is_violation(label):
    return "NoHelmet" in label   # bất kỳ rider nào (lái hoặc ngồi sau) không đội mũ
