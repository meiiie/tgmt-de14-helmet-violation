"""Chạy thử probe thật trên ảnh và realtime (lặp ảnh tĩnh, không đo chất lượng video)."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(4)
import app


def main():
    sample = next((ROOT / "data/helmet/frames").glob("*/*.jpg"))
    frame = cv2.imread(str(sample))
    assert frame is not None
    frame = cv2.resize(frame, (1920, 1080))
    assert "DINOv3" in app.DEFAULT_BACKEND, app.DEFAULT_BACKEND
    print("Default backend:", app.DEFAULT_BACKEND, flush=True)
    result = app.analyze(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                         None, app.BACKENDS[1], 3, 8, 3, 640, True)
    assert result[0] is not None and Path(result[5]).is_file()
    print("Offline image + CSV: PASS", result[2], flush=True)
    rt = app._rt_pipeline("yolo26s", 640, 5, True)
    try:
        for i in range(12):
            rt.step(frame, i)
        drained = rt.drain(timeout=120)
        assert drained and rt.err is None, rt.err
        assert rt.n_cls > 0 and rt.votes, "Không có crop đầu được phân loại"
        assert all(np.isfinite(v) and 0 <= v <= 1 for vs in rt.votes.values() for v in vs)
        report = {"default_backend": app.DEFAULT_BACKEND, "offline_summary": result[2],
                  "offline_csv_exists": Path(result[5]).is_file(),
                  "realtime_smoke": "12 repeated still frames, not a video accuracy or FPS evaluation",
                  "realtime_frames": rt.n_frames, "classified_crops": rt.n_cls,
                  "tracks_with_votes": len(rt.votes), "drained": drained, "worker_error": rt.err}
        out = ROOT / "outputs/metrics/local_dino_smoke.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Realtime YOLO26s + DINOv3 worker: PASS", flush=True)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    finally:
        rt.close()


if __name__ == "__main__":
    main()
