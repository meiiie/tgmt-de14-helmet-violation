"""Đề 14 – chế độ REALTIME.
YOLO (YOLO26s hoặc YOLO11s, COCO person+motorcycle) chạy MỌI frame (fp16 trên GPU) -> ghép rider–xe -> ByteTrack
-> phân loại đầu DINOv3-S + logistic probe chạy BẤT ĐỒNG BỘ trong worker thread (hàng đợi, gom batch, fp16),
chỉ cho track mới hoặc mỗi N frame/track -> bỏ phiếu theo track -> F2 tính online (tăng dần theo track).
SAM 3.1 vẫn là chế độ offline độ chính xác cao (src/de14/pipeline.py)."""
from __future__ import annotations
import threading, queue, time
from collections import defaultdict
import numpy as np, cv2
from .core import associate, head_box, crop, decide, ioa, rider_zone, keep_track, VIOL, OK, UNDET
from .tracking import ByteTrackWrapper

MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)

class FastDinoProbe:
    """DINOv3 probe (joblib từ src/train_probe.py), embedding fp16 trên GPU, tiền xử lý tự viết (resize 224, chuẩn hoá ImageNet)."""
    def __init__(self, probe_path, device=None, half=True, size=224):
        import joblib, torch
        from transformers import AutoModel
        from .classifiers import DinoEmbedder
        import os
        d = joblib.load(probe_path); self.clf = d["clf"]; mid = d["model_id"]
        loc = os.path.join(os.environ.get("DINO_LOCAL_DIR", "/nonexistent"), mid.split("/")[-1])
        if os.path.isdir(loc): mid = loc
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu"); self.half = half and self.dev.startswith("cuda")
        self.m = AutoModel.from_pretrained(mid).to(self.dev).eval()
        if self.half: self.m = self.m.half()
        self.torch, self.size = torch, size
    def __call__(self, crops):
        if not crops: return []
        t = self.torch
        x = np.stack([(cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2RGB), (self.size, self.size), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255 - MEAN) / STD for c in crops])
        x = t.from_numpy(x).permute(0, 3, 1, 2).to(self.dev); x = x.half() if self.half else x
        with t.inference_mode():
            o = self.m(pixel_values=x)
            f = o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state[:, 0]
            f = t.nn.functional.normalize(f.float(), dim=-1).cpu().numpy()
        return self.clf.predict_proba(f)[:, 1].tolist()

class OnlineTrackStats:
    """F2 online: cập nhật tăng dần person_score (trung bình IoA người–vùng rider) và disp_norm (span tâm xe / đường chéo TB)."""
    __slots__ = ("n", "cov", "xmin", "xmax", "ymin", "ymax", "diag")
    def __init__(self): self.n = 0; self.cov = 0.0; self.xmin = self.ymin = 1e9; self.xmax = self.ymax = -1e9; self.diag = 0.0
    def update(self, m, riders):
        cx, cy = (m[0]+m[2])/2, (m[1]+m[3])/2; self.n += 1
        self.cov += max((ioa(p, rider_zone(m)) for p in riders), default=0.0); self.diag += float(np.hypot(m[2]-m[0], m[3]-m[1]))
        self.xmin, self.xmax, self.ymin, self.ymax = min(self.xmin, cx), max(self.xmax, cx), min(self.ymin, cy), max(self.ymax, cy)
    def get(self):
        if not self.n: return {"person_score": 0.0, "disp_norm": 0.0, "n_frames": 0}
        span = float(np.hypot(self.xmax - self.xmin, self.ymax - self.ymin))
        return {"person_score": self.cov / self.n, "disp_norm": span / max(1e-6, self.diag / self.n), "n_frames": self.n}

class RealtimePipeline:
    def __init__(self, yolo_weights="weights/yolo26s.pt", probe="weights/dino_probe.joblib", imgsz=960, conf=0.15, device=None,
                 half=True, every_n=5, batch=32, n_min=3, hi=0.6, lo=0.4, min_head=8, fps=10, track_filter=None, classifier=None):
        import torch
        from ultralytics import YOLO
        self.dev = device or ("cuda:0" if torch.cuda.is_available() else "cpu"); self.half = half and self.dev.startswith("cuda")
        self.yolo = YOLO(yolo_weights); self.imgsz, self.conf = imgsz, conf
        self.cls = classifier or FastDinoProbe(probe, device=self.dev, half=self.half)
        self.tracker = ByteTrackWrapper(fps=fps); self.every_n, self.batch = every_n, batch
        self.n_min, self.hi, self.lo, self.min_head = n_min, hi, lo, min_head
        self.track_filter = track_filter
        self.votes = defaultdict(list); self.stats = defaultdict(OnlineTrackStats); self.last_job = {}
        self.lock = threading.Lock(); self.q = queue.Queue(); self.n_frames = 0; self.n_cls = 0; self.t_cls = 0.0
        self.err = None; self.stop = False; self.w = threading.Thread(target=self._worker, daemon=True); self.w.start()
    # ---- worker: gom batch các job (tid, [crop...]) -> 1 lần forward DINOv3 ----
    def _worker(self):
        while not self.stop:
            try: jobs = [self.q.get(timeout=0.05)]
            except queue.Empty: continue
            while len(jobs) < self.batch:
                try: jobs.append(self.q.get_nowait())
                except queue.Empty: break
            crops = [c for _, cs in jobs for c in cs]; t = time.time()
            try: ps = self.cls(crops)
            except Exception as e:   # không để worker chết -> q.join() treo
                self.err = repr(e); print("[realtime worker] lỗi:", self.err, flush=True); ps = [None] * len(crops)
            self.t_cls += time.time() - t; self.n_cls += len(crops); k = 0
            with self.lock:
                for tid, cs in jobs:
                    p = ps[k:k+len(cs)]; k += len(cs)
                    p = [x for x in p if x is not None]
                    if p: self.votes[tid].append(max(p))   # rider "tệ nhất" (lái + ngồi sau)
            for _ in jobs: self.q.task_done()
    def drain(self, timeout=120):
        """Chờ worker xử lý hết hàng đợi, có hạn chót (không treo vô hạn)."""
        t = time.time()
        while self.q.unfinished_tasks and time.time() - t < timeout and self.w.is_alive(): time.sleep(0.01)
        return self.q.unfinished_tasks == 0
    def close(self): self.drain(); self.stop = True; self.w.join(timeout=2)
    def decision(self, tid):
        with self.lock: v = list(self.votes[tid])
        st = self.stats[tid].get()
        if self.track_filter and not keep_track(st, **self.track_filter): return "filtered", st
        return decide(v, self.n_min, self.hi, self.lo)[0], st
    def step(self, img, fi=None):
        """Xử lý 1 frame. Trả về list (tid, box_union, moto_box)."""
        self.n_frames += 1
        r = self.yolo.predict(img, imgsz=self.imgsz, conf=self.conf, classes=[0, 3], half=self.half, device=self.dev, verbose=False)[0]
        b = r.boxes.xyxy.cpu().numpy(); c = r.boxes.cls.cpu().numpy().astype(int); s = r.boxes.conf.cpu().numpy()
        motos = b[c == 3].tolist(); ms = s[c == 3].tolist(); persons = b[c == 0].tolist()
        assoc = associate(persons, motos); ub = []
        for mi, m in enumerate(motos):
            bs = [m] + [persons[i] for i in assoc[mi]]
            ub.append([min(x[0] for x in bs), min(x[1] for x in bs), max(x[2] for x in bs), max(x[3] for x in bs)])
        out = []
        for tid, tb, idx in self.tracker.update(ub, ms, img.shape):
            if idx < 0 or idx >= len(motos): continue
            m = motos[idx]; riders = [persons[i] for i in assoc[idx]]
            self.stats[tid].update(m, riders)
            if tid not in self.last_job or self.n_frames - self.last_job[tid] >= self.every_n:
                rs = riders or [[m[0], m[1] - (m[3]-m[1]), m[2], m[3]]]
                cs = [x for x in (crop(img, head_box(p), self.min_head) for p in rs) if x is not None]
                if cs: self.q.put((tid, [x.copy() for x in cs])); self.last_job[tid] = self.n_frames
            out.append((tid, tb, m))
        return out
    def draw(self, img, recs, fps=None):
        col = {VIOL: (0, 0, 255), OK: (0, 200, 0), UNDET: (0, 200, 255), "filtered": (128, 128, 128)}
        for tid, tb, _ in recs:
            lab, _ = self.decision(tid); x1, y1, x2, y2 = map(int, tb)
            cv2.rectangle(img, (x1, y1), (x2, y2), col[lab], 2)
            cv2.putText(img, f"#{tid} {dict([(VIOL,'VIOL'),(OK,'OK'),(UNDET,'?'),('filtered','F2')])[lab]}", (x1, max(12, y1-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col[lab], 2)
        if fps is not None:
            H = img.shape[0]; cv2.rectangle(img, (0, H - 34), (260, H), (0, 0, 0), -1)
            cv2.putText(img, f"FPS {fps:5.1f}  q={self.q.qsize()}", (6, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return img
