"""Bộ phân loại mũ trên crop đầu.
- YoloHelmetCls: YOLO helmet pretrained (nnsohamnn / iam-tsr) chạy trên crop → p(không mũ).
- DinoProbe: embedding DINOv3 (gated, lõi) + logistic regression / kNN. KHÔNG có fallback DINOv2: tải lỗi -> raise (báo 'chưa xác định').
"""
from __future__ import annotations
import os
import numpy as np, cv2

DINOV3_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"   # gated

class YoloHelmetCls:
    def __init__(self, weights="weights/helmet_yolo11s_nnsohamnn.pt", conf=0.5, imgsz=160, device=None):
        from ultralytics import YOLO
        self.m = YOLO(weights); self.conf, self.imgsz = conf, imgsz
        self.device = device
        names = {k: v.lower().replace("_", " ").replace("-", " ") for k, v in self.m.names.items()}
        self.no_ids = {k for k, v in names.items() if "without" in v or "no helmet" in v or "nohelmet" in v}
        self.yes_ids = {k for k, v in names.items() if "helmet" in v and k not in self.no_ids}
    def __call__(self, crops):
        ps = []
        for c in crops:
            r = self.m.predict(c, imgsz=self.imgsz, conf=self.conf, verbose=False, device=self.device)[0]
            if len(r.boxes) == 0:
                ps.append(None); continue
            valid = [i for i in range(len(r.boxes)) if float(r.boxes.conf[i]) >= max(.5, self.conf)
                     and int(r.boxes.cls[i]) in self.no_ids | self.yes_ids]
            if not valid:
                ps.append(None); continue
            k = max(valid, key=lambda i: float(r.boxes.conf[i])); cls = int(r.boxes.cls[k]); cf = float(r.boxes.conf[k])
            ps.append(cf if cls in self.no_ids else 1 - cf)
        return ps


class ClipHelmetVerifier:
    """Semantic cross-check for fabric/fur/sun hats. Returns a signed similarity margin,
    not a calibrated probability. Positive means evidence for a non-protective head covering.
    """
    PROMPTS = (
        "a photo of a person wearing a motorcycle helmet.",
        "a photo of a person wearing a protective crash helmet.",
        "a photo of a person with a bare head and no helmet.",
        "a photo of a person wearing a soft fabric cap, not a helmet.",
        "a photo of a person wearing a winter fur hat, not a helmet.",
        "a photo of a child wearing a sun hat, not a helmet.",
    )

    def __init__(self, model_path="weights/clip-vit-base-patch32", device="cpu", cache_text=False):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(model_path, local_files_only=True).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
        self.device, self.torch = device, torch
        self.cache_text, self._text_features = cache_text, None
        self.stats = {"text_forwards": 0, "image_forwards": 0}

    def __call__(self, crops):
        if not crops: return []
        images = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops]
        with self.torch.inference_mode():
            if self.cache_text:
                if self._text_features is None:
                    tokens = self.proc(text=list(self.PROMPTS), return_tensors="pt", padding=True).to(self.device)
                    text = self.model.get_text_features(**tokens)
                    text = getattr(text, "pooler_output", text)
                    self._text_features = text / self.torch.linalg.vector_norm(text, dim=-1, keepdim=True)
                    self.stats["text_forwards"] += 1
                pixels = self.proc(images=images, return_tensors="pt").to(self.device)
                features = self.model.get_image_features(**pixels)
                features = getattr(features, "pooler_output", features)
                features = features / self.torch.linalg.vector_norm(features, dim=-1, keepdim=True)
                scores = ((self._text_features @ features.T) * self.model.logit_scale.exp()).T.float()
            else:
                inputs = self.proc(text=list(self.PROMPTS), images=images, return_tensors="pt", padding=True).to(self.device)
                scores = self.model(**inputs).logits_per_image.float()
                self.stats["text_forwards"] += 1
            self.stats["image_forwards"] += 1
        return (scores[:, 2:].max(dim=1).values - scores[:, :2].max(dim=1).values).cpu().tolist()


class VerifiedHelmetCls:
    """Keep the trained classifier, check non-protective hats, abstain on conflict/tiny heads.

    Thresholds are conservative heuristics; they require validation on held-out street
    footage. Two user regression images are never used to fit either model.
    """
    def __init__(self, classifier, verifier, min_side=32, batch_size=32):
        if batch_size < 1: raise ValueError("batch_size must be positive")
        self.base, self.verifier, self.min_side = classifier, verifier, min_side
        self.batch_size = batch_size

    def __call__(self, crops):
        out = [None] * len(crops)
        valid = [i for i, c in enumerate(crops) if min(c.shape[:2]) >= self.min_side]
        if not valid: return out
        for start in range(0, len(valid), self.batch_size):
            indices = valid[start:start+self.batch_size]
            cs = [crops[i] for i in indices]
            probs = self.base(cs)
            if self.verifier is None:
                if len(probs) != len(cs): raise ValueError("Classifier output count differs from crop count")
                for i, p in zip(indices, probs): out[i] = p
                continue
            margins = self.verifier(cs)
            if len(probs) != len(cs) or len(margins) != len(cs):
                raise ValueError("Classifier/verifier output count differs from crop count")
            for i, p, margin in zip(indices, probs, margins):
                if margin >= 2.:
                    out[i] = float(1. / (1. + np.exp(-margin)))
                elif p is not None and p >= .6 and margin > -1.:
                    out[i] = p
                elif p is not None and p <= .4 and margin <= -.5:
                    out[i] = p
        return out

class DinoEmbedder:
    def __init__(self, model_id=None, device=None, local_files_only=False):
        import torch
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        from transformers import AutoImageProcessor, AutoModel
        mid = model_id or DINOV3_ID
        canonical_id = mid
        loc = os.path.join(os.environ.get("DINO_LOCAL_DIR", "/nonexistent"), mid.split("/")[-1])
        if os.path.isdir(loc): mid = loc   # checkpoint từ private Kaggle dataset (không công khai trọng số gated)
        self.proc = AutoImageProcessor.from_pretrained(mid, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(mid, local_files_only=local_files_only).to(device).eval()
        self.model_id, self.loaded_from, self.device, self.torch = canonical_id, mid, device, torch
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
    def load(cls, path, device=None, local_files_only=False):
        import joblib; d = joblib.load(path); p = cls(DinoEmbedder(d["model_id"], device, local_files_only), d["kind"]); p.clf = d["clf"]; return p
