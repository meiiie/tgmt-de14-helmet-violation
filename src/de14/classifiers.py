"""Bộ phân loại mũ trên crop đầu.
- YoloHelmetCls: YOLO helmet pretrained (nnsohamnn / iam-tsr) chạy trên crop → p(không mũ).
- DinoProbe: embedding DINOv3 (gated, lõi) hoặc DINOv2 (fallback không gated, CHỈ để smoke test CPU) + logistic regression / kNN.
"""
from __future__ import annotations
import os
import numpy as np, cv2

DINOV3_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"   # gated
DINOV2_FALLBACK = "facebook/dinov2-small"                  # Apache-2.0, không gated

class YoloHelmetCls:
    def __init__(self, weights="weights/helmet_yolo11s_nnsohamnn.pt", conf=0.15, imgsz=160):
        from ultralytics import YOLO
        self.m = YOLO(weights); self.conf, self.imgsz = conf, imgsz
        self.no_ids = {k for k, v in self.m.names.items() if "without" in v.lower() or "no" in v.lower().split("_")}
    def __call__(self, crops):
        ps = []
        for c in crops:
            r = self.m.predict(c, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
            if len(r.boxes) == 0:
                ps.append(None); continue
            k = int(r.boxes.conf.argmax()); cls = int(r.boxes.cls[k]); cf = float(r.boxes.conf[k])
            ps.append(cf if cls in self.no_ids else 1 - cf)
        return ps

class DinoEmbedder:
    def __init__(self, model_id=None, device="cpu"):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        mid = model_id or (DINOV3_ID if os.environ.get("HF_TOKEN") else DINOV2_FALLBACK)
        try:
            self.proc = AutoImageProcessor.from_pretrained(mid); self.model = AutoModel.from_pretrained(mid).to(device).eval()
        except Exception as e:
            if mid == DINOV2_FALLBACK: raise
            print(f"[fallback] {mid} không tải được ({type(e).__name__}); dùng {DINOV2_FALLBACK} (CHỈ smoke test)")
            mid = DINOV2_FALLBACK
            self.proc = AutoImageProcessor.from_pretrained(mid); self.model = AutoModel.from_pretrained(mid).to(device).eval()
        self.model_id, self.device, self.torch = mid, device, torch
    def __call__(self, crops, bs=32):
        feats = []
        for i in range(0, len(crops), bs):
            ims = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops[i:i+bs]]
            x = self.proc(images=ims, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                o = self.model(**x)
            f = o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state[:, 0]
            feats.append(self.torch.nn.functional.normalize(f, dim=-1).cpu().numpy())
        return np.concatenate(feats) if feats else np.zeros((0, 1))

class DinoProbe:
    """Linear probe (logistic regression) trên embedding DINO; tuỳ chọn kNN."""
    def __init__(self, embedder: DinoEmbedder, kind="linear"):
        self.e, self.kind, self.clf = embedder, kind, None
    def fit(self, crops, y):
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        X = self.e(crops)
        self.clf = (LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced") if self.kind == "linear"
                    else KNeighborsClassifier(n_neighbors=7, weights="distance"))
        self.clf.fit(X, y); return self
    def __call__(self, crops):
        if not crops: return []
        return self.clf.predict_proba(self.e(crops))[:, 1].tolist()
    def save(self, path):
        import joblib; joblib.dump({"clf": self.clf, "model_id": self.e.model_id, "kind": self.kind}, path)
    @classmethod
    def load(cls, path, device="cpu"):
        import joblib; d = joblib.load(path); p = cls(DinoEmbedder(d["model_id"], device), d["kind"]); p.clf = d["clf"]; return p
