"""Pipeline đầu-cuối Đề 14.
Frame -> detector (SAM 3.1 | YOLO11) -> ghép rider–xe (lái + ngồi sau) -> ByteTrack theo xe -> crop đầu từng rider
-> classifier (DINO probe | YOLO helmet) -> bỏ phiếu theo track -> nhãn {không đội mũ, có đội mũ, chưa xác định}
Đầu ra: CSV track, CSV quan sát, ảnh bằng chứng (crop + timestamp), video chú thích."""
from __future__ import annotations
import os, csv, json, time
from collections import defaultdict
import numpy as np, cv2
from .core import track_stats, keep_track, associate, head_box, crop, decide, ioa, UNDET, VIOL, OK
from .tracking import ByteTrackWrapper

COLORS = {VIOL: (0, 0, 255), OK: (0, 200, 0), UNDET: (0, 200, 255)}

def heads_for_riders(riders, sam_heads):
    """Nếu có box 'head' từ SAM 3.1 nằm trong nửa trên rider -> dùng; ngược lại heuristic."""
    out = []
    for p in riders:
        top = [p[0], p[1], p[2], p[1] + 0.4 * (p[3] - p[1])]
        cands = [h for h in sam_heads if ioa(h, top) > 0.6]
        out.append(max(cands, key=lambda h: (h[2]-h[0])*(h[3]-h[1])) if cands else head_box(p))
    return out

class Pipeline:
    def __init__(self, detector, classifier, fps=10, n_min=3, hi=0.6, lo=0.4, min_head=8, conf_moto=0.3, track_filter=None):
        self.track_filter = track_filter   # None hoặc dict(tau_person, tau_disp, tau_person_static) – xem core.keep_track
        self.det, self.cls, self.fps = detector, classifier, fps
        self.n_min, self.hi, self.lo, self.min_head, self.conf_moto = n_min, hi, lo, min_head, conf_moto

    def run(self, frames, out_dir=None, video_name="video", write_video=True, evidence=True):
        """frames: iterable (frame_idx, img_bgr). Trả về dict kết quả."""
        tracker = ByteTrackWrapper(fps=self.fps)
        obs = defaultdict(list); per_frame = []; best_ev = {}; geo = defaultdict(list)
        vw = None; t0 = time.time(); n = 0; t_det = t_cls = 0.0
        if out_dir:
            os.makedirs(os.path.join(out_dir, "evidence"), exist_ok=True)
        for fi, img in frames:
            n += 1; t = time.time(); d = self.det(img); t_det += time.time() - t
            ms = [(b, s) for b, s in zip(d["motorcycle"], d.get("motorcycle_score", [1]*len(d["motorcycle"]))) if s >= self.conf_moto]
            motos = [b for b, _ in ms]; persons = d["person"]
            assoc = associate(persons, motos)
            # box theo dõi = hợp của xe + rider (ổn định hơn)
            ub = []
            for mi, m in enumerate(motos):
                bs = [m] + [persons[i] for i in assoc[mi]]
                ub.append([min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)])
            tracks = tracker.update(ub, [s for _, s in ms], img.shape)
            frame_rec = []
            for tid, tb, idx in tracks:
                if idx < 0 or idx >= len(motos): continue
                riders = [persons[i] for i in assoc[idx]]
                geo[tid].append((motos[idx], riders))   # cho bộ lọc track (người trên xe, độ dịch chuyển)
                if not riders:
                    # rider không được phát hiện riêng -> vùng rider suy từ bbox xe (kéo lên trên 1 chiều cao xe)
                    m = motos[idx]; h = m[3] - m[1]
                    riders = [[m[0], m[1] - 1.0 * h, m[2], m[3]]]
                hbs = heads_for_riders(riders, d.get("head", []))
                crops = [crop(img, h, self.min_head) for h in hbs]
                valid = [(c, h) for c, h in zip(crops, hbs) if c is not None]
                t = time.time(); ps = self.cls([c for c, _ in valid]) if valid else []; t_cls += time.time() - t
                ps_ok = [(p, c, h) for p, (c, h) in zip(ps, valid) if p is not None]
                if ps_ok:
                    pmax, cmax, hmax = max(ps_ok, key=lambda x: x[0])   # rider "tệ nhất" trên xe (lái + ngồi sau)
                    obs[tid].append({"frame": fi, "t": fi / self.fps, "p_no_helmet": pmax, "n_riders": len(riders)})
                    key = (tid, pmax >= 0.5)
                    if evidence and (key not in best_ev or abs(pmax - 0.5) > abs(best_ev[key][0] - 0.5)):
                        best_ev[key] = (pmax, fi, cmax.copy())
                frame_rec.append((tid, tb, ps_ok[0][0] if ps_ok else None, [(h, p) for p, (c, h) in zip(ps, valid)] + [(h, None) for c_, h in zip(crops, hbs) if c_ is None]))
            per_frame.append((fi, frame_rec))
            if out_dir and write_video:
                if vw is None:
                    vw = cv2.VideoWriter(os.path.join(out_dir, f"{video_name}_annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (img.shape[1], img.shape[0]))
                vis = img.copy()
                for tid, tb, p, hbs in frame_rec:
                    lab = decide([o["p_no_helmet"] for o in obs[tid]], self.n_min, self.hi, self.lo)[0]
                    c = COLORS[lab]; x1, y1, x2, y2 = map(int, tb)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)
                    cv2.putText(vis, f"#{tid} {'VIOL' if lab==VIOL else 'OK' if lab==OK else '?'}", (x1, max(10, y1-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
                    for h, p in hbs:
                        hc = (0, 200, 255) if p is None else ((0, 0, 255) if p >= 0.5 else (0, 200, 0))
                        cv2.rectangle(vis, tuple(map(int, h[:2])), tuple(map(int, h[2:])), hc, 2)
                        cv2.putText(vis, "?" if p is None else f"{p:.2f}", (int(h[0]), int(h[3]) + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, hc, 1)
                cv2.putText(vis, f"t={fi/self.fps:.1f}s", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                vw.write(vis)
        if vw is not None: vw.release()
        self.last_vis = vis if (out_dir and write_video and n) else None
        el = time.time() - t0
        results = {}
        for tid in {t for _, fr in per_frame for t, *_ in fr}:
            ps = [o["p_no_helmet"] for o in obs[tid]]
            lab, r, k = decide(ps, self.n_min, self.hi, self.lo)
            fr = [fi for fi, rec in per_frame if any(t == tid for t, *_ in rec)]
            st = track_stats(geo[tid])
            if self.track_filter and not keep_track(st, **self.track_filter): continue
            results[tid] = {"track_id": tid, "decision": lab, "no_helmet_ratio": r, "n_obs": k,
                            "t_start": min(fr) / self.fps, "t_end": max(fr) / self.fps, **st}
        if out_dir:
            with open(os.path.join(out_dir, f"{video_name}_tracks.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, ["track_id", "decision", "no_helmet_ratio", "n_obs", "t_start", "t_end", "person_score", "disp_norm", "n_frames", "evidence"]); w.writeheader()
                for tid, r in sorted(results.items()):
                    ev = ""
                    key = (tid, r["decision"] != OK)
                    if evidence and key in best_ev:
                        p, fi, c = best_ev[key]
                        ev = f"evidence/{video_name}_track{tid}_t{fi/self.fps:06.2f}s_p{p:.2f}.jpg"
                        cv2.imwrite(os.path.join(out_dir, ev), cv2.resize(c, (max(64, c.shape[1]*4), max(64, c.shape[0]*4)), interpolation=cv2.INTER_NEAREST))
                    w.writerow({**r, "evidence": ev})
            with open(os.path.join(out_dir, f"{video_name}_observations.csv"), "w", newline="") as f:
                w = csv.writer(f); w.writerow(["track_id", "frame", "t_sec", "p_no_helmet", "n_riders"])
                for tid, os_ in obs.items():
                    for o in os_: w.writerow([tid, o["frame"], f"{o['t']:.2f}", f"{o['p_no_helmet']:.3f}", o["n_riders"]])
        return {"tracks": results, "per_frame": per_frame, "frames": n, "sec": el,
                "fps": n / el if el else 0, "t_det": t_det, "t_cls": t_cls}
