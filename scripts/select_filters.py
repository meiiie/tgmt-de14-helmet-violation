"""Chọn ngưỡng bộ lọc track trên 1 fold (theo clip) và đánh giá trên các clip còn lại.
Fold tinh chỉnh = clip test sắp xếp theo tên, lấy chỉ số 0,3,6 (3 clip); đánh giá = 4 clip còn lại.
Tiêu chí chọn: F1 = 2PR/(P+R) với P tính CẢ track không khớp GT, R trên mọi GT track vi phạm.
Dùng: python scripts/select_filters.py <metrics_dir> <out.json>"""
import sys, json, glob, os, itertools
sys.path.insert(0, "src")
from de14.core import keep_track, VIOL, OK, UNDET
from de14.helmet_gt import is_violation

def score(tracks, gts, clips, flt):
    tp = fp = fpu = und = miss = pos = 0; matched = 0
    for c in clips:
        per = {}
        kept = [t for t in tracks if t["clip"] == c and keep_track(t, **flt)]
        for t in kept:
            if t["gt"] is None: fpu += t["decision"] == VIOL
            else: per.setdefault(t["gt"], []).append(t["decision"])
        for g, lab in gts[c].items():
            v = is_violation(lab); pos += v; ds = per.get(g)
            if ds is None: miss += 1; continue
            matched += 1; d = VIOL if VIOL in ds else OK if OK in ds else UNDET
            if d == UNDET: und += 1
            elif d == VIOL: tp += v; fp += (not v)
    P = tp / max(1, tp + fp); Pu = tp / max(1, tp + fp + fpu); R = tp / max(1, pos)
    return {"precision": P, "precision_incl_unmatched": Pu, "recall": R, "undetermined_rate": und / max(1, matched),
            "f1_incl_unmatched": 2 * Pu * R / max(1e-9, Pu + R), "TP": tp, "FP": fp, "FP_unmatched": fpu, "missed_gt": miss}

TP_ = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]
TD_ = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]
TS_ = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
GRID = {"none": [{}],
        "f1_person": [{"tau_person": a} for a in TP_],
        "f2_stationary": [{"tau_disp": d, "tau_person_static": s} for d in TD_ for s in TS_],
        "f1+f2": [{"tau_person": a, "tau_disp": d, "tau_person_static": s} for a in TP_ for d in TD_ for s in TS_ if s >= a]}

if __name__ == "__main__":
    md, out = sys.argv[1], sys.argv[2]; res = {}
    for f in sorted(glob.glob(f"{md}/core_v4_*.json")):
        d = json.load(open(f)); tr, gts = d["_tracks"], d["_gt"]
        clips = sorted(gts); tune = clips[0::3]; ev = [c for c in clips if c not in tune]
        fps = sum(d["per_clip_frames"][c] for c in ev) / max(1e-6, sum(d["per_clip_sec"][c] for c in ev))
        name = os.path.basename(f)[8:-5]; res[name] = {"tune_clips": tune, "eval_clips": ev, "fps_end2end_eval": fps}
        for k, grid in GRID.items():
            best = max(grid, key=lambda g: (round(score(tr, gts, tune, g)["f1_incl_unmatched"], 6), -len(g), -sum(g.values())))
            res[name][k] = {"thresholds": best, "tune": score(tr, gts, tune, best), "eval": score(tr, gts, ev, best)}
            e = res[name][k]["eval"]
            print(f"{name:28s} {k:14s} {json.dumps(best):70s} P={e['precision']:.3f} Pu={e['precision_incl_unmatched']:.3f} R={e['recall']:.3f} U={e['undetermined_rate']:.3f} F1u={e['f1_incl_unmatched']:.3f} FPS={fps:.2f}")
    json.dump(res, open(out, "w"), indent=1)
