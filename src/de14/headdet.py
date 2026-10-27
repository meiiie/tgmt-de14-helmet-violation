"""Tầng phát hiện đầu (tuỳ chọn) cho chế độ realtime: DEIMv2-Wholebody49 (PINTO0309, Apache-2.0, ONNX) lớp Head (id 7)
chạy TRONG crop rider (xe + người, mở rộng), thay cho vùng đầu suy từ box người.
I/O theo demo của PINTO (demo/demo_hfhpe_onnx.py): resize thẳng về HxW cố định, BGR->RGB, /255 (HGNetV2) hoặc ImageNet (DINOv3),
đầu ra `label_xyxy_score` [1,Q,6] = (class, x1, y1, x2, y2, score), toạ độ chuẩn hoá nếu không có input orig_target_sizes."""
from __future__ import annotations
import time
import numpy as np, cv2
HEAD = 7
MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]; STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]

class DeimHeadDetector:
    def __init__(self, onnx_path, device="cuda", min_score=0.1, require_gpu=False):
        import onnxruntime as ort
        if device.startswith("cuda") and hasattr(ort, "preload_dlls"):
            try: ort.preload_dlls()   # nạp cuDNN/cuBLAS CUDA 12 từ wheel nvidia-* mà torch đã cài
            except Exception as e: print("ort.preload_dlls:", e, flush=True)
        prov = (["CUDAExecutionProvider", "CPUExecutionProvider"] if device.startswith("cuda") else ["CPUExecutionProvider"])
        self.s = ort.InferenceSession(onnx_path, providers=prov); self.providers = self.s.get_providers()
        if require_gpu and self.providers[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"head detector KHÔNG chạy trên GPU: providers={self.providers}")
        i = self.s.get_inputs()[0]; self.inp = i.name; self.hw = (int(i.shape[2]), int(i.shape[3]))
        self.names = {x.name for x in self.s.get_inputs()}; self.norm = "imagenet" if "dinov3" in onnx_path.lower() else "div255"
        self.min_score = min_score; self.t = 0.0; self.n = 0
    def __call__(self, img):
        """-> list (score, [x1,y1,x2,y2]) các đầu trong ảnh (pixel), sắp theo score giảm dần."""
        H, W = img.shape[:2]; h, w = self.hw
        x = cv2.cvtColor(cv2.resize(img, (w, h)), cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0
        if self.norm == "imagenet": x = (x - MEAN) / STD
        feed = {self.inp: x[None]}
        if "orig_target_sizes" in self.names: feed["orig_target_sizes"] = np.array([[W, H]], np.float32)
        t = time.perf_counter(); (p,) = self.s.run(["label_xyxy_score"], feed); self.t += time.perf_counter() - t; self.n += 1
        p = p[0]; p = p[(p[:, 0].astype(int) == HEAD) & (p[:, 5] >= self.min_score)]
        out = []
        for _, x1, y1, x2, y2, sc in p:
            if "orig_target_sizes" not in self.names: x1, x2, y1, y2 = x1 * W, x2 * W, y1 * H, y2 * H
            if x2 - x1 >= 2 and y2 - y1 >= 2: out.append((float(sc), [float(x1), float(y1), float(x2), float(y2)]))
        return sorted(out, key=lambda z: -z[0])

def square(b, margin=0.05):
    x1, y1, x2, y2 = b; w, h = x2 - x1, y2 - y1; x1 -= w * margin; x2 += w * margin; y1 -= h * margin; y2 += h * margin
    s = max(x2 - x1, y2 - y1); cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return [cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2]
