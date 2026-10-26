"""Kiểm tra: lỗi trong classifier của worker KHÔNG làm treo vòng lặp chính / drain()."""
import sys, time, glob, cv2
sys.path.insert(0, "src")
from de14.realtime import RealtimePipeline
class Boom:
    def __init__(self): self.n = 0
    def __call__(self, crops):
        self.n += 1
        if self.n % 2: raise RuntimeError("giả lập lỗi fp16 DINOv3")
        return [0.9] * len(crops)
pl = RealtimePipeline("weights/yolo11s.pt", None, imgsz=640, device="cpu", every_n=2, classifier=Boom())
t = time.time()
for f in sorted(glob.glob("data/helmet/frames/Bago_urban_3/*.jpg"))[:30]:
    pl.step(cv2.resize(cv2.imread(f), (1920, 1080)))
ok = pl.drain(timeout=30); pl.close()
print(f"frames {pl.n_frames}, drained={ok}, worker_alive_after_errors={pl.err is not None}, err={pl.err}, votes={sum(len(v) for v in pl.votes.values())}, {time.time()-t:.1f}s")
assert ok and pl.err, "FAIL"; print("PASS")
