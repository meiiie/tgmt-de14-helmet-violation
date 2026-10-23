"""Backend phát hiện: (1) SAM 3.1 text-prompt (lõi, cần HF_TOKEN + GPU), (2) YOLO11 COCO (baseline / fallback CPU)."""
from __future__ import annotations
import os, json
import numpy as np

class YoloDetector:
    name = "yolo11s-coco"
    def __init__(self, weights="weights/yolo11s.pt", imgsz=1280, conf=0.25, device="cpu"):
        from ultralytics import YOLO
        self.m = YOLO(weights); self.imgsz, self.conf, self.device = imgsz, conf, device
    def __call__(self, img):
        r = self.m.predict(img, imgsz=self.imgsz, conf=self.conf, classes=[0, 3], verbose=False, device=self.device)[0]
        b = r.boxes.xyxy.cpu().numpy(); c = r.boxes.cls.cpu().numpy().astype(int); s = r.boxes.conf.cpu().numpy()
        return {"person": b[c == 0].tolist(), "motorcycle": b[c == 3].tolist(),
                "person_score": s[c == 0].tolist(), "motorcycle_score": s[c == 3].tolist(), "head": []}

class Sam3Detector:
    """SAM 3.1 (facebook/sam3.1, gated) — ảnh đơn: prompt văn bản 'motorcycle', 'person', 'head', 'helmet'.
    Dùng repo chính thức https://github.com/facebookresearch/sam3 (pip install -e .).
    Checkpoint SAM 3.1 'sam3.1_multiplex.pt' tải qua HF_TOKEN. Image model dùng build_sam3_image_model(checkpoint_path=...)."""
    name = "sam3.1"
    PROMPTS = ("motorcycle", "person", "head", "helmet")
    def __init__(self, version="sam3.1", conf=0.4, device="cuda"):
        if not os.environ.get("HF_TOKEN"):
            raise RuntimeError("SAM 3.1 là model gated: cần HF_TOKEN đã được Meta chấp thuận truy cập facebook/sam3.1")
        import sys, os as _os
        sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
        from sam31_cache import load
        self.proc = load(device); self.proc.confidence_threshold = conf; self.device = device
    def __call__(self, img_bgr):
        from PIL import Image
        import torch
        im = Image.fromarray(img_bgr[:, :, ::-1])
        out = {"person": [], "motorcycle": [], "head": [], "helmet": []}
        with torch.autocast(self.device, dtype=torch.bfloat16), torch.inference_mode():
            st = self.proc.set_image(im)
            for p in self.PROMPTS:
                o = self.proc.set_text_prompt(state=st, prompt=p)
                out[p] = o["boxes"].float().cpu().numpy().tolist()
                out[p + "_score"] = o["scores"].float().cpu().numpy().tolist()
        return out

def sam31_video_tracks(video_path, prompts=("motorcycle", "person"), max_num_objects=64):
    """SAM 3.1 Object Multiplex: phát hiện + phân đoạn + theo vết toàn video bằng prompt văn bản.
    Trả về {prompt: {frame_idx: [(obj_id, xyxy_rel)]}}. Cần GPU + HF_TOKEN."""
    from sam3.model_builder import build_sam3_predictor
    pred = build_sam3_predictor(version="sam3.1", max_num_objects=max_num_objects, use_fa3=False)
    res = {}
    for p in prompts:
        sid = pred.handle_request({"type": "start_session", "resource_path": video_path})["session_id"]
        pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0, "text": p})
        res[p] = {}
        for out in pred.handle_stream_request({"type": "propagate_in_video", "session_id": sid}):
            f = out["frame_index"]; o = out["outputs"]
            res[p][f] = [(int(i), [x, y, x + w, y + h]) for i, (x, y, w, h) in zip(o["out_obj_ids"], o["out_boxes_xywh"])]
        pred.handle_request({"type": "close_session", "session_id": sid})
    return res

class CachedSam3Detector:
    """Đọc kết quả SAM 3.1 đã chạy sẵn (src/sam31_cache.py) theo tên frame — để chạy lại pipeline/đánh giá nhanh."""
    name = "sam3.1(cached)"
    def __init__(self, cache_json, conf=0.4):
        self.c = json.load(open(cache_json)); self.conf = conf; self.key = None
    def __call__(self, img):
        rec = self.c[self.key]; out = {}
        for p in ("motorcycle", "person", "head"):
            r = rec.get(p, {"boxes": [], "scores": []})
            keep = [i for i, s in enumerate(r["scores"]) if s >= self.conf]
            out[p] = [r["boxes"][i] for i in keep]; out[p + "_score"] = [r["scores"][i] for i in keep]
        return out

def build_detector(kind="auto", **kw):
    if kind == "sam3.1" or (kind == "auto" and os.environ.get("HF_TOKEN")):
        try:
            return Sam3Detector(**kw)
        except Exception as e:
            if kind == "sam3.1":
                raise
            print(f"[fallback] SAM 3.1 không khả dụng ({e}); dùng YOLO11 COCO (CPU smoke test)")
    return YoloDetector()
