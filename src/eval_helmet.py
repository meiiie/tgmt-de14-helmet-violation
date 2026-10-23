"""Đánh giá theo TRACK trên HELMET test split (Myanmar). GT dương = xe có ít nhất 1 rider (lái/ngồi sau) không đội mũ.
Dùng: python src/eval_helmet.py --method b1 --detector yolo --classifier yolo --clips Bago_urban_3 ... --stride 1"""
import argparse, glob, json, os, sys, time
from collections import defaultdict, Counter
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from de14.core import iou, VIOL, OK, UNDET
from de14.helmet_gt import load_clip, is_violation

FRAMES = os.environ.get("HELMET_FRAMES", "data/helmet/frames")

def frames_of(clip, stride, only=None):
    fs = sorted(glob.glob(f"{FRAMES}/{clip}/*.jpg"))
    for f in fs[::stride]:
        k = os.path.basename(f)
        if only is not None and k not in only: continue
        img = cv2.imread(f)
        if img is None: continue
        if img.shape[:2] != (1080, 1920):  # bản HF 640x640 bị kéo giãn -> khôi phục 16:9 gốc
            img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_CUBIC)
        yield int(k[:-4]), k, img

def match_tracks(per_frame, gt_frames, thr=0.3):
    votes = defaultdict(Counter)
    for fi, rec in per_frame:
        g = gt_frames.get(fi, [])
        for tid, tb, *_ in rec:
            best = max(g, key=lambda x: iou(tb, x[1]), default=None)
            if best is not None and iou(tb, best[1]) >= thr: votes[tid][best[0]] += 1
            else: votes[tid][None] += 1
    return {t: c.most_common(1)[0][0] for t, c in votes.items()}

def score(gt_tracks_by_clip, pred_by_clip):
    """pred_by_clip: {clip: {gt_tid: decision}}, plus unmatched FP list"""
    tp = fp = fn_ok = und = miss = pos = neg = tn = 0; fp_unmatched = 0
    for clip, gts in gt_tracks_by_clip.items():
        P = pred_by_clip[clip]
        for g, lab in gts.items():
            v = is_violation(lab); pos += v; neg += (not v)
            d = P["gt"].get(g)
            if d is None: miss += 1; continue
            if d == UNDET: und += 1
            elif d == VIOL: tp += v; fp += (not v)
            else: fn_ok += v; tn += (not v)
        fp_unmatched += P["fp_unmatched"]
    matched = pos + neg - miss
    return {"gt_tracks": pos + neg, "gt_violation_tracks": pos, "matched": matched, "missed": miss,
            "TP": tp, "FP_on_helmet_tracks": fp, "FP_unmatched": fp_unmatched, "FN_decided_ok": fn_ok, "TN": tn, "undetermined": und,
            "precision": tp / max(1, tp + fp), "precision_incl_unmatched": tp / max(1, tp + fp + fp_unmatched),
            "recall_all_gt": tp / max(1, pos), "recall_decided": tp / max(1, tp + fn_ok),
            "undetermined_rate": und / max(1, matched), "coverage": matched / max(1, pos + neg)}

def run_b1(args, clips):
    from de14.pipeline import Pipeline
    from de14.detectors import YoloDetector, CachedSam3Detector
    from de14.classifiers import YoloHelmetCls, DinoProbe
    cls = YoloHelmetCls(args.helmet_weights) if args.classifier == "yolo" else DinoProbe.load(args.probe)
    gt_tracks, pred, tot_f, tot_s, tdet, tcls = {}, {}, 0, 0, 0, 0
    for clip in clips:
        gtf, gtt = load_clip(clip, W=1920, H=1080)
        only = None
        if args.detector == "yolo": det = YoloDetector(imgsz=1280, conf=0.15)
        else:
            det = CachedSam3Detector(f"outputs/sam31/{clip}.json", conf=args.sam_conf); only = set(det.c)
        fps_eff = 10 / args.stride
        pl = Pipeline(det, cls, fps=fps_eff, n_min=args.n_min, min_head=args.min_head)
        gen = list(frames_of(clip, args.stride, only))
        if not gen: continue
        def it():
            for fi, k, img in gen:
                if hasattr(det, "key"): det.key = k
                yield fi, img
        out_dir = os.path.join(args.out, clip)
        r = pl.run(it(), out_dir=out_dir, video_name=clip, write_video=True)
        frames_used = {fi for fi, _, _ in gen}
        # GT track chỉ tính nếu xuất hiện trong frame đã xử lý
        gtt = {g: l for g, l in gtt.items() if any(any(x[0] == g for x in gtf.get(fi, [])) for fi in frames_used)}
        m = match_tracks(r["per_frame"], gtf)
        per_gt = defaultdict(list); fpu = 0
        for tid, tr in r["tracks"].items():
            g = m.get(tid)
            if g is None: fpu += tr["decision"] == VIOL
            else: per_gt[g].append(tr["decision"])
        dec = {g: (VIOL if VIOL in ds else OK if OK in ds else UNDET) for g, ds in per_gt.items()}
        gt_tracks[clip] = gtt; pred[clip] = {"gt": dec, "fp_unmatched": fpu}
        tot_f += r["frames"]; tot_s += r["sec"]; tdet += r["t_det"]; tcls += r["t_cls"]
        print(clip, "frames", r["frames"], f"fps {r['fps']:.2f}", "pred tracks", len(r["tracks"]), "gt tracks", len(gtt), flush=True)
    res = score(gt_tracks, pred); res.update({"frames": tot_f, "fps_end2end": tot_f / max(1e-6, tot_s),
                                              "sec_det": tdet, "sec_cls": tcls})
    return res

def run_b0(args, clips):
    """B0: YOLO helmet trên TOÀN khung, từng frame độc lập; gán box helmet vào GT xe chứa tâm; không tracking.
    Track-level cho B0 = bỏ phiếu đa số các quyết định frame của cùng GT track (oracle association) để so sánh."""
    from ultralytics import YOLO
    m = YOLO(args.helmet_weights); no_ids = {k for k, v in m.names.items() if "without" in v.lower()}
    gt_tracks, pred = {}, {}; tf = ts = 0; fr_stats = Counter()
    for clip in clips:
        gtf, gtt = load_clip(clip, W=1920, H=1080); per = defaultdict(list); used = set()
        for fi, k, img in frames_of(clip, args.stride):
            used.add(fi); t = time.time()
            r = m.predict(img, imgsz=1280, conf=0.25, verbose=False)[0]; ts += time.time() - t; tf += 1
            bx = r.boxes.xyxy.numpy(); cl = r.boxes.cls.numpy().astype(int)
            for g, gb, lab in gtf.get(fi, []):
                inside = [c for b, c in zip(bx, cl) if gb[0] <= (b[0]+b[2])/2 <= gb[2] and gb[1] <= (b[1]+b[3])/2 <= gb[3]]
                d = UNDET if not inside else (VIOL if any(c in no_ids for c in inside) else OK)
                per[g].append(d)
                fr_stats[(is_violation(lab), d)] += 1
        gtt = {g: l for g, l in gtt.items() if g in per}
        dec = {}
        for g, ds in per.items():
            c = Counter(x for x in ds if x != UNDET)
            dec[g] = UNDET if not c else c.most_common(1)[0][0]
        gt_tracks[clip] = gtt; pred[clip] = {"gt": dec, "fp_unmatched": 0}
    res = score(gt_tracks, pred)
    tp = fr_stats[(True, VIOL)]; fp = fr_stats[(False, VIOL)]; fn = fr_stats[(True, OK)]
    und = sum(v for (g, d), v in fr_stats.items() if d == UNDET); n = sum(fr_stats.values())
    res.update({"frame_level": {"precision": tp / max(1, tp+fp), "recall_all": tp / max(1, sum(v for (g, d), v in fr_stats.items() if g)),
                                "undetermined_rate": und / max(1, n), "n_moto_frames": n},
                "frames": tf, "fps_end2end": tf / max(1e-6, ts)})
    return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="b1"); ap.add_argument("--detector", default="yolo")
    ap.add_argument("--classifier", default="yolo"); ap.add_argument("--probe", default="weights/dino_probe.joblib")
    ap.add_argument("--helmet_weights", default="weights/helmet_yolo11s_nnsohamnn.pt")
    ap.add_argument("--clips", nargs="+", default=["Bago_highway_10", "Bago_highway_20", "Bago_rural_3", "Bago_rural_49", "Bago_urban_3", "Bago_urban_50"])
    ap.add_argument("--stride", type=int, default=1); ap.add_argument("--n_min", type=int, default=3)
    ap.add_argument("--min_head", type=int, default=8); ap.add_argument("--sam_conf", type=float, default=0.4)
    ap.add_argument("--out", default="outputs/eval"); ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    res = run_b0(a, a.clips) if a.method == "b0" else run_b1(a, a.clips)
    tag = a.tag or f"{a.method}_{a.detector}_{a.classifier}_s{a.stride}"
    os.makedirs("outputs/metrics", exist_ok=True)
    json.dump(res, open(f"outputs/metrics/{tag}.json", "w"), indent=2, ensure_ascii=False)
    print(tag, json.dumps(res, indent=1, ensure_ascii=False))
