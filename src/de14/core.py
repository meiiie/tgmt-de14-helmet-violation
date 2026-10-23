"""Đề 14 – lõi dùng chung: hình học, ghép người–xe, crop đầu, bỏ phiếu theo track."""
from __future__ import annotations
import numpy as np

UNDET = "chưa xác định"; VIOL = "không đội mũ"; OK = "có đội mũ"

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i / u if u > 0 else 0.0

def ioa(a, b):
    """phần diện tích a nằm trong b"""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max(1e-6, (a[2]-a[0])*(a[3]-a[1]))

def associate(persons, motos, min_ioa=0.25):
    """Gán mỗi person cho xe máy phù hợp nhất (người lái + người ngồi sau).
    Điều kiện: chân (đáy bbox người) nằm trong bbox xe mở rộng ngang và người chồng lấn xe.
    Trả về dict moto_idx -> list person_idx, sắp xếp theo x (vị trí người trên xe)."""
    out = {i: [] for i in range(len(motos))}
    for pi, p in enumerate(persons):
        best, bs = None, 0.0
        cx, by = (p[0]+p[2]) / 2, p[3]
        for mi, m in enumerate(motos):
            mw, mh = m[2]-m[0], m[3]-m[1]
            if not (m[0]-0.25*mw <= cx <= m[2]+0.25*mw and m[1]-0.1*mh <= by <= m[3]+0.35*mh):
                continue
            s = ioa(p, [m[0]-0.25*mw, m[1]-mh, m[2]+0.25*mw, m[3]+0.2*mh]) + iou(p, m)
            if s > bs and ioa(p, m) + 1e-6 >= 0 and s >= min_ioa:
                best, bs = mi, s
        if best is not None:
            out[best].append(pi)
    for k in out:
        out[k].sort(key=lambda i: (persons[i][0]+persons[i][2]) / 2)
    return out

def head_box(p, frac=0.28):
    """Ước lượng vùng đầu = phần trên bbox người (vuông hoá, mở rộng 15%)."""
    x1, y1, x2, y2 = p; w, h = x2-x1, y2-y1
    hh = h*frac; cx = (x1+x2)/2; s = max(hh, min(w, hh*1.3)) * 1.15
    return [cx - s/2, y1 - 0.08*hh, cx + s/2, y1 - 0.08*hh + s]

def crop(img, b, min_side=0):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in b]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2-x1 < max(2, min_side) or y2-y1 < max(2, min_side):
        return None
    return img[y1:y2, x1:x2]

def decide(probs, n_min=3, hi=0.6, lo=0.4):
    """probs: list xác suất 'không mũ' của từng quan sát hợp lệ. Trả (nhãn, tỉ lệ, n)."""
    if len(probs) < n_min:
        return UNDET, (float(np.mean(probs)) if probs else None), len(probs)
    r = float(np.mean([p >= 0.5 for p in probs]))
    lab = VIOL if r >= hi else (OK if r <= lo else UNDET)
    return lab, r, len(probs)
