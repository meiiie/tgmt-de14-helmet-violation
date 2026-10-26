"""Đánh giá chế độ realtime trên HELMET test (mọi frame, stride 1), cùng luật khớp track/chấm điểm như offline.
FPS end-to-end = số frame / thời gian thực (đọc ảnh + YOLO + ghép + ByteTrack + hàng đợi DINOv3 + chờ worker xong). Bỏ 1 frame warm-up.
python src/eval_realtime.py --weights weights/yolo26s.pt --imgsz 960 --probe weights/dino_probe.joblib --tag rt_yolo26s_960"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from eval_helmet import frames_of, match_tracks
from de14.helmet_gt import load_clip
from de14.realtime import RealtimePipeline
from select_filters import score
CLIPS = ["Bago_highway_10", "Bago_highway_20", "Bago_rural_3", "Bago_rural_49", "Bago_urban_3", "Bago_urban_50", "Mandalay_1_105"]
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--weights", default="weights/yolo26s.pt"); ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--probe", default="weights/dino_probe.joblib"); ap.add_argument("--every_n", type=int, default=5)
    ap.add_argument("--clips", nargs="+", default=CLIPS); ap.add_argument("--stride", type=int, default=1); ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--device", default=None); ap.add_argument("--no_half", action="store_true"); ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    flt = json.load(open("configs/track_filter.json"))["track_filter"]
    tracks, gts, tot_f, tot_s, per = [], {}, 0, 0.0, {}
    warm = None
    for clip in a.clips:
        if not os.path.isdir(os.path.join(os.environ.get("HELMET_FRAMES", "data/helmet/frames"), clip)): print("skip", clip); continue
        gen = list(frames_of(clip, a.stride))
        if a.max_frames: gen = gen[:a.max_frames]
        pl = RealtimePipeline(a.weights, a.probe, imgsz=a.imgsz, device=a.device, half=not a.no_half, every_n=a.every_n, fps=10 / a.stride)
        pl.step(gen[0][2].copy()); pl.drain()  # warm-up (không tính), rồi reset tracker/phiếu
        pl.tracker = type(pl.tracker)(fps=10 / a.stride); pl.votes.clear(); pl.stats.clear(); pl.last_job.clear(); pl.n_frames = 0
        gtf, gtt = load_clip(clip, W=1920, H=1080); per_frame = []
        t0 = time.time()
        for fi, k, img in gen:
            recs = pl.step(img, fi); per_frame.append((fi, [(tid, tb) for tid, tb, _ in recs]))
        pl.drain(); el = time.time() - t0
        used = {fi for fi, _, _ in gen}
        gts[clip] = {g: l for g, l in gtt.items() if any(any(x[0] == g for x in gtf.get(fi, [])) for fi in used)}
        m = match_tracks(per_frame, gtf)
        for tid in {t for _, r in per_frame for t, _ in r}:
            lab, st = pl.decision(tid) if False else (None, pl.stats[tid].get())
            from de14.core import decide
            with pl.lock: v = list(pl.votes[tid])
            tracks.append({"clip": clip, "track_id": tid, "decision": decide(v, pl.n_min, pl.hi, pl.lo)[0], "gt": m.get(tid), **st})
        per[clip] = {"frames": len(gen), "sec": el, "fps": len(gen) / el, "dino_crops": pl.n_cls, "dino_sec": pl.t_cls}
        tot_f += len(gen); tot_s += el; pl.close()
        print(clip, per[clip], flush=True)
    clips = sorted(gts); tune = [c for c in clips if c in ("Bago_highway_10", "Bago_rural_49", "Mandalay_1_105")]; ev = [c for c in clips if c not in tune]
    res = {"tag": a.tag, "weights": os.path.basename(a.weights), "imgsz": a.imgsz, "every_n": a.every_n, "half": not a.no_half, "stride": a.stride,
           "fps_end2end": tot_f / max(1e-6, tot_s), "fps_eval4": sum(per[c]["frames"] for c in ev) / max(1e-6, sum(per[c]["sec"] for c in ev)),
           "per_clip": per, "track_filter": flt}
    for nm, cl in [("all", clips), ("eval4", ev)]:
        res[nm] = {"none": score(tracks, gts, cl, {}), "f2": score(tracks, gts, cl, flt)}
    res["_tracks"] = tracks; res["_gt"] = gts
    os.makedirs("outputs/metrics", exist_ok=True); json.dump(res, open(f"outputs/metrics/{a.tag}.json", "w"), indent=1, ensure_ascii=False)
    for nm in ("all", "eval4"):
        for k in ("none", "f2"):
            e = res[nm][k]; print(a.tag, nm, k, {x: round(e[x], 3) for x in ("precision", "precision_incl_unmatched", "recall", "undetermined_rate", "f1_incl_unmatched")})
    print(a.tag, "FPS all %.2f eval4 %.2f" % (res["fps_end2end"], res["fps_eval4"]))
