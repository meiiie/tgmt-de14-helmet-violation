"""ByteTrack (bản cài đặt trong Ultralytics) dùng độc lập cho mọi backend phát hiện (YOLO hoặc SAM 3.1)."""
import numpy as np
from types import SimpleNamespace

class ByteTrackWrapper:
    def __init__(self, fps=10, high=0.4, low=0.1, new=0.45, buffer=30, match=0.8, fuse_score=True):
        from ultralytics.trackers.byte_tracker import BYTETracker
        args = SimpleNamespace(track_high_thresh=high, track_low_thresh=low, new_track_thresh=new,
                               track_buffer=buffer, match_thresh=match, fuse_score=fuse_score, frame_rate=fps)
        self.t = BYTETracker(args)
        self.update_fps = fps
        self.tentative_measurements = []

    def snapshot(self):
        """Installed ByteTrack ages tracks in update units, not original frames."""
        return {"update_index": self.t.frame_id,
                "confirmed_ids": [int(t.track_id) for t in self.t.tracked_stracks if t.is_activated],
                "tentative_ids": [int(t.track_id) for t in self.t.tracked_stracks if not t.is_activated],
                "lost_ids": [int(t.track_id) for t in self.t.lost_stracks],
                "removed_ids": [int(t.track_id) for t in self.t.removed_stracks],
                "lost_buffer_updates": self.t.max_frames_lost,
                "lost_buffer_seconds": self.t.max_frames_lost/self.update_fps}
    def update(self, boxes, scores, shape):
        from ultralytics.engine.results import Boxes
        if len(boxes) == 0:
            data = np.zeros((0, 6), dtype=np.float32)
        else:
            data = np.concatenate([np.asarray(boxes, np.float32), np.asarray(scores, np.float32)[:, None],
                                   np.zeros((len(boxes), 1), np.float32)], 1)
        out = self.t.update(Boxes(data, shape[:2]))
        # Expose only detections measured in THIS update, never Kalman predictions.
        self.tentative_measurements = [
            (int(t.track_id), list(boxes[int(t.idx)]), int(t.idx))
            for t in self.t.tracked_stracks
            if not t.is_activated and t.frame_id == self.t.frame_id and 0 <= int(t.idx) < len(boxes)]
        # out: [x1,y1,x2,y2,track_id,score,cls,idx]
        return [(int(r[4]), r[:4].tolist(), int(r[7])) for r in out]
