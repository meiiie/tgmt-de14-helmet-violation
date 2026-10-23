"""ByteTrack (bản cài đặt trong Ultralytics) dùng độc lập cho mọi backend phát hiện (YOLO hoặc SAM 3.1)."""
import numpy as np
from types import SimpleNamespace

class ByteTrackWrapper:
    def __init__(self, fps=10, high=0.4, low=0.1, new=0.45, buffer=30, match=0.8):
        from ultralytics.trackers.byte_tracker import BYTETracker
        args = SimpleNamespace(track_high_thresh=high, track_low_thresh=low, new_track_thresh=new,
                               track_buffer=buffer, match_thresh=match, fuse_score=True)
        self.t = BYTETracker(args, frame_rate=fps)
    def update(self, boxes, scores, shape):
        from ultralytics.engine.results import Boxes
        if len(boxes) == 0:
            data = np.zeros((0, 6), dtype=np.float32)
        else:
            data = np.concatenate([np.asarray(boxes, np.float32), np.asarray(scores, np.float32)[:, None],
                                   np.zeros((len(boxes), 1), np.float32)], 1)
        out = self.t.update(Boxes(data, shape[:2]))
        # out: [x1,y1,x2,y2,track_id,score,cls,idx]
        return [(int(r[4]), r[:4].tolist(), int(r[7])) for r in out]
