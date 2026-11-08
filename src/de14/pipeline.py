"""Pipeline đầu-cuối Đề 14.
Frame -> detector (SAM 3.1 | YOLO11) -> ghép rider–xe (lái + ngồi sau) -> ByteTrack theo xe -> crop đầu từng rider
-> classifier (DINO probe | YOLO helmet) -> bỏ phiếu theo track -> nhãn {không đội mũ, có đội mũ, chưa xác định}
Đầu ra: CSV track, CSV quan sát, ảnh bằng chứng (crop + timestamp), video chú thích."""
from __future__ import annotations
import os, csv, json, time
from collections import defaultdict
import numpy as np, cv2
from .core import track_stats, keep_track, associate, head_box, crop, decide, ioa, iou, rider_zone, merge_motorcycle_fragments, UNDET, VIOL, OK
from .tracking import ByteTrackWrapper
from .headdet import square

COLORS = {VIOL: (0, 0, 255), OK: (0, 200, 0), UNDET: (0, 200, 255)}


def draw_annotations(img, records, decisions, timestamp_s, kept_tracks=None, hi=.6, lo=.4, untracked=()):
    """Draw only measured heads and causal votes; no tracker/classifier updates."""
    vis = img.copy()
    for tid, tb, _, hbs in records:
        if kept_tracks is not None and tid not in kept_tracks: continue
        lab = decisions.get(tid, UNDET)
        c = COLORS[lab]; x1, y1, x2, y2 = map(int, tb)
        cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)
        cv2.putText(vis, f"#{tid} {'VIOL' if lab==VIOL else 'OK' if lab==OK else '?'} riders={len(hbs)}",
                    (x1, max(10, y1-4)), cv2.FONT_HERSHEY_SIMPLEX, .45, c, 1)
        for h, p in hbs:
            head_label = decide([p], 1, hi, lo)[0]
            hc = COLORS[head_label]
            cv2.rectangle(vis, tuple(map(int, h[:2])), tuple(map(int, h[2:])), hc, 2)
            text = '?' if head_label == UNDET else ('NO HELMET' if head_label == VIOL else 'HELMET')
            cv2.putText(vis, text, (int(h[0]), int(h[3])+12), cv2.FONT_HERSHEY_SIMPLEX, .35, hc, 1)
    for detection in untracked:
        box = detection['box']
        color = COLORS[UNDET]
        cv2.rectangle(vis, tuple(map(int, box[:2])), tuple(map(int, box[2:])), color, 1)
        cv2.putText(vis, f'UNCONFIRMED riders={len(detection["heads"])}',
                    (int(box[0]), max(10, int(box[1]) - 4)), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
        for head, probability in detection['heads']:
            label = decide([probability], 1, hi, lo)[0]
            cv2.rectangle(vis, tuple(map(int, head[:2])), tuple(map(int, head[2:])), COLORS[label], 2)
            text = '?' if label == UNDET else 'NO HELMET*' if label == VIOL else 'HELMET*'
            cv2.putText(vis, text, (int(head[0]), int(head[3]) + 12), cv2.FONT_HERSHEY_SIMPLEX, .35, COLORS[label], 1)
    cv2.putText(vis, f't={timestamp_s:.1f}s', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    return vis

def heads_for_riders(riders, sam_heads):
    """Nếu có box 'head' từ SAM 3.1 nằm trong nửa trên rider -> dùng; ngược lại heuristic."""
    out = []; used = set()
    for p in riders:
        top = [p[0], p[1], p[2], p[1] + 0.4 * (p[3] - p[1])]
        cands = [i for i, h in enumerate(sam_heads) if i not in used and ioa(h, top) > 0.6]
        if cands:
            i = min(cands, key=lambda i: abs((sam_heads[i][0]+sam_heads[i][2]-p[0]-p[2])/2))
            used.add(i); out.append(sam_heads[i])
        else:
            out.append(head_box(p))
    return out


def heads_for_motorcycles(motos, heads, persons=None, audit=None):
    """Assign each independently detected head once, including occluded passengers/children.

    Counting does not depend on full-body detections. Geometry is only an association
    cue, not proof that every nearby pedestrian is a rider.
    """
    out = {i: [] for i in range(len(motos))}
    unique = []
    for index, h in enumerate(heads):
        if any(iou(h, other) > .5 for _, other in unique):
            if audit is not None: audit.append({"index": index, "box": h, "owner": None, "reason": "duplicate_head"})
        else: unique.append((index, h))
    # A background pedestrian's upper body must not become a passenger merely
    # because its head lies in the tall, expanded zone of a foreground bike.
    body_assoc = associate(persons or [], motos, min_foot_fraction=.25)
    body_owner = {pi: mi for mi, people in body_assoc.items() for pi in people}
    for index, h in unique:
        cx, cy = (h[0]+h[2])/2, (h[1]+h[3])/2
        body_candidates = []
        for pi, p in enumerate(persons or []):
            if (p[0] <= cx <= p[2] and p[1]-.15*(h[3]-h[1]) <= cy <= p[1]+.4*(p[3]-p[1])
                    and ioa(h, p) > .3):
                distance = abs(cx-(p[0]+p[2])/2)/max(p[2]-p[0], 1) + abs(cy-(p[1]+.12*(p[3]-p[1])))/max(p[3]-p[1], 1)
                body_candidates.append((distance, pi))
        if body_candidates:
            pi = min(body_candidates)[1]; owner = body_owner.get(pi)
            if owner is not None: out[owner].append(h)
            if audit is not None:
                audit.append({"index": index, "box": h, "owner": owner, "person_index": pi,
                              "reason": "associated_body" if owner is not None else "body_outside_rider_zone"})
            continue
        candidates = []
        for i, m in enumerate(motos):
            z = rider_zone(m); w, height = m[2]-m[0], m[3]-m[1]
            if persons is not None:
                support = [m] + [persons[pi] for pi in body_assoc[i]]
                # Allow head-only children within the supported rider envelope;
                # do not absorb unrelated heads in the extra horizontal margin.
                z = [min(b[0] for b in support), z[1], max(b[2] for b in support), z[3]]
            if z[0] <= cx <= z[2] and z[1] <= cy <= m[1] + .35*height:
                distance = abs(cx-(m[0]+m[2])/2)/max(w, 1) + .2*abs(cy-m[1])/max(height, 1)
                candidates.append((distance, i))
        owner = min(candidates)[1] if candidates else None
        if owner is not None: out[owner].append(h)
        if audit is not None:
            audit.append({"index": index, "box": h, "owner": owner,
                          "reason": "head_only_geometry" if owner is not None else "outside_rider_zone"})
    return out


def vehicle_probability(probs, hi=.6, lo=.4):
    """One clear violation suffices; 'all helmeted' requires evidence for every head."""
    known = [p for p in probs if p is not None and np.isfinite(p)]
    if any(p >= hi for p in known): return max(known)
    if probs and len(known) == len(probs) and all(p <= lo for p in known): return max(known)
    return None

class Pipeline:
    def __init__(self, detector, classifier, fps=10, n_min=3, hi=0.6, lo=0.4, min_head=8, conf_moto=0.3, track_filter=None, batch_heads=False,
                 backfill_tentative=True, tracker_fuse_score=True, retain_untracked_heads=False):
        self.track_filter = track_filter   # None hoặc dict(tau_person, tau_disp, tau_person_static) – xem core.keep_track
        self.det, self.cls, self.fps = detector, classifier, fps
        self.batch_heads = batch_heads
        self.backfill_tentative, self.tracker_fuse_score = backfill_tentative, tracker_fuse_score
        self.retain_untracked_heads = retain_untracked_heads
        self.n_min, self.hi, self.lo, self.min_head, self.conf_moto = n_min, hi, lo, min_head, conf_moto

    def run(self, frames, out_dir=None, video_name="video", write_video=True, evidence=True, trace=None, tracking=True, render=False, frame_metadata=None):
        """frames: iterable (frame_idx, img_bgr). Trả về dict kết quả."""
        tracker = ByteTrackWrapper(fps=self.fps, fuse_score=self.tracker_fuse_score)
        obs = defaultdict(list); per_frame = []; best_ev = {}; geo = defaultdict(list); rider_counts = defaultdict(int)
        # Only the offline second rendering pass can rewrite past frames.
        backfill = self.backfill_tentative and tracking and not (render or (out_dir and write_video))
        pending = {}; frame_records = {}; backfill_events = []; discarded_pending = 0; skipped_headless = 0
        untracked_detections = {}; deferred_predictions = {}; reused_predictions = 0
        vw = None; t0 = time.time(); n = 0; t_det = t_cls = 0.0
        classifier_calls = 0; frame_latency_ms = []; timeline = {}; frame_decisions = {}; tracking_audit = []
        counters = dict(person_detections=0, head_detections=0, associated_heads=0, missing_head_frames=0, invalid_crops=0)
        presence = defaultdict(lambda: dict(person_positive_frames=0, head_positive_frames=0, invalid_crops=0))
        if out_dir:
            os.makedirs(os.path.join(out_dir, "evidence"), exist_ok=True)
        for fi, img in frames:
            meta = (frame_metadata or {}).get(fi, {})
            timeline[fi] = {"inference_index": fi, "source_index": meta.get("source_index", fi),
                            "timestamp_s": meta.get("timestamp_s", fi/self.fps)}
            timestamp = timeline[fi]["timestamp_s"]
            frame_records[fi] = []; per_frame.append((fi, frame_records[fi]))
            if not tracking and n:
                raise ValueError("tracking=False is for one static image; use tracking=True for video")
            frame_start = time.perf_counter()
            n += 1; t = time.time(); d = self.det(img); t_det += time.time() - t
            counters["person_detections"] += len(d["person"])
            counters["head_detections"] += len(d.get("head", []))
            if trace: trace({"frame": fi, "stage": "detector_output", "detections": d})
            if d.get("head_detection", bool(d.get("head"))):
                boxes, scores, groups = merge_motorcycle_fragments(
                    d["motorcycle"], d.get("motorcycle_score", [1]*len(d["motorcycle"])),
                    d["person"], d.get("person_score", [1]*len(d["person"])), tracker.t.args.new_track_thresh)
                if trace: trace({"frame": fi, "stage": "vehicle_consolidation", "boxes": boxes, "scores": scores, "source_indices": groups})
                d = {**d, "motorcycle": boxes, "motorcycle_score": scores}
            ms = [(b, s) for b, s in zip(d["motorcycle"], d.get("motorcycle_score", [1]*len(d["motorcycle"]))) if s >= self.conf_moto]
            if trace:
                scores = d.get("motorcycle_score", [1]*len(d["motorcycle"]))
                trace({"frame": fi, "stage": "motorcycle_confidence", "boxes": d["motorcycle"], "scores": scores,
                       "threshold": self.conf_moto, "kept_indices": [i for i, s in enumerate(scores) if s >= self.conf_moto],
                       "rejected": [{"index": i, "score": s, "reason": "below_conf_moto"} for i, s in enumerate(scores) if s < self.conf_moto]})
            motos = [b for b, _ in ms]; persons = d["person"]
            assoc = associate(persons, motos)
            head_audit = [] if trace else None
            detected_heads = heads_for_motorcycles(motos, d.get("head", []), persons=persons, audit=head_audit)
            if trace:
                trace({"frame": fi, "stage": "person_association", "motorcycles": motos, "persons": persons, "owners": assoc})
                trace({"frame": fi, "stage": "head_association", "motorcycles": motos, "owners": detected_heads, "decisions": head_audit})
            has_head_detector = d.get("head_detection", bool(d.get("head")))
            # box theo dõi = hợp của xe + rider (ổn định hơn)
            ub = []
            for mi, m in enumerate(motos):
                bs = [m] + [persons[i] for i in assoc[mi]]
                ub.append([min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)])
            # A static image has no temporal evidence to confirm. Its detections
            # have already passed conf_moto; do not discard them at ByteTrack's
            # high/new-track gates intended for video.
            static_rejected = []
            if tracking:
                tracks = tracker.update(ub, [s for _, s in ms], img.shape)
            else:
                tracks = []
                for i, box in enumerate(ub):
                    # Without temporal confirmation, admit a weak vehicle only
                    # when its independently detected rider/body supports it.
                    if ms[i][1] < tracker.t.args.new_track_thresh and not (detected_heads[i] or assoc[i]):
                        static_rejected.append({"index": i, "reason": "weak_static_box_without_rider_evidence"})
                    else: tracks.append((i+1, box, i))
            if trace:
                tracked_indices = {idx for _, _, idx in tracks}
                trace({"frame": fi, "stage": "tracking", "input_boxes": ub, "input_scores": [s for _, s in ms],
                       "tracks": tracks, "untracked_indices": [i for i in range(len(motos)) if i not in tracked_indices],
                       "mode": "video" if tracking else "static_image",
                       "static_rejected": static_rejected,
                       "high_threshold": tracker.t.args.track_high_thresh if tracking else None,
                       "new_track_threshold": tracker.t.args.new_track_thresh if tracking else None})
            if tracking:
                tracking_audit.append({**timeline[fi], **tracker.snapshot()})
            frame_rec = frame_records[fi]; track_items = []; item_frames = []
            tentative = tracker.tentative_measurements if backfill else []
            tentative_ids = {tid for tid, _, _ in tentative}
            measured_samples = []; measured_crops = {}
            for tid, tb, idx in tracks + tentative:
                if idx < 0 or idx >= len(motos): continue
                riders = [persons[i] for i in assoc[idx]]
                hbs = detected_heads[idx] if has_head_detector else heads_for_riders(riders, [])
                crops = [crop(img, square(h, margin=.15) if has_head_detector else h, self.min_head) for h in hbs]
                if tid in tentative_ids:
                    # Decoders may reuse the image array; retain small crop copies only.
                    crops = [c.copy() if c is not None else None for c in crops]
                sample = (fi, tid, tb, motos[idx], riders, hbs, crops, has_head_detector)
                measured_crops[idx] = (hbs, crops)
                if tid in tentative_ids:
                    # Recover helmet evidence, not extra unsupported parked-bike boxes.
                    # A valid crop may classify UNKNOWN; eligibility is geometric only.
                    if any(c is not None for c in crops):
                        pending[tid] = [sample]
                    else:
                        skipped_headless += 1
                else:
                    previous = pending.pop(tid, [])
                    measured_samples.extend(previous)
                    backfill_events.extend({"track_id": tid, "frame": s[0],
                        "source_index": timeline[s[0]]["source_index"], "confirmed_at_frame": fi,
                        "confirmed_at_source_index": timeline[fi]["source_index"]} for s in previous)
                    measured_samples.append(sample)
            for tid in list(pending):
                if tid not in tentative_ids:
                    discarded_pending += len(pending.pop(tid))
            for observed_fi, tid, tb, moto, riders, hbs, crops, sample_has_heads in measured_samples:
                if observed_fi in untracked_detections:
                    untracked_detections[observed_fi] = [detection for detection in untracked_detections[observed_fi]
                                                       if detection['candidate_track_id'] != tid]
                # An independently detected head is rider-presence evidence even
                # when the person's body is occluded (e.g. a child at a red light).
                geo[tid].append((moto, riders + hbs if sample_has_heads else riders))
                rider_counts[tid] = max(rider_counts[tid], len(hbs))
                valid = [(c, h) for c, h in zip(crops, hbs) if c is not None]
                presence[tid]["person_positive_frames"] += bool(riders)
                presence[tid]["head_positive_frames"] += bool(hbs)
                presence[tid]["invalid_crops"] += len(crops)-len(valid)
                counters["associated_heads"] += len(hbs)
                counters["missing_head_frames"] += not bool(hbs)
                counters["invalid_crops"] += len(crops)-len(valid)
                track_items.append((tid, tb, hbs, crops, valid))
                item_frames.append(observed_fi)
            if trace:
                for observed_fi in sorted(set(item_frames) | {fi}):
                    trace({"frame": observed_fi, "processing_frame": fi, "stage": "crops", "heads": [
                        {"track_id": tid, "box": h, "crop_box": square(h, margin=.15) if sample[-1] else h,
                         "crop_shape": list(c.shape) if c is not None else None, "status": "valid" if c is not None else "too_small"}
                        for sample, ofi, (tid, _, hbs, crops, _) in zip(measured_samples, item_frames, track_items) if ofi == observed_fi for h, c in zip(hbs, crops)]})
            provisional_items = []; provisional_indices = []
            untracked_detections[fi] = []
            if self.retain_untracked_heads and tracking and has_head_detector:
                tracked_indices = {idx for _, _, idx in tracks}
                for idx, box in enumerate(ub):
                    if idx in tracked_indices or not detected_heads[idx]:
                        continue
                    if idx in measured_crops:
                        hbs, crops = measured_crops[idx]
                    else:
                        hbs = detected_heads[idx]
                        crops = [crop(img, square(head, margin=.15), self.min_head) for head in hbs]
                    valid = [(c, h) for c, h in zip(crops, hbs) if c is not None]
                    provisional_items.append((None, box, hbs, crops, valid))
                    provisional_indices.append(idx)
            predictions = []
            if self.retain_untracked_heads:
                all_items = track_items + provisional_items
                all_frames = item_frames + [fi] * len(provisional_items)
                resolved = dict(deferred_predictions)
                requests = {}
                for observed_fi, (_, _, _, _, valid) in zip(all_frames, all_items):
                    for c, h in valid:
                        key = (observed_fi, tuple(h))
                        if key not in resolved:
                            requests.setdefault(key, c)
                        else:
                            reused_predictions += 1
                if self.batch_heads:
                    flat = list(requests.values())
                    started = time.perf_counter()
                    values = self.cls(flat) if flat else []
                    t_cls += time.perf_counter() - started
                    classifier_calls += bool(flat)
                    if len(values) != len(flat):
                        raise ValueError('Classifier output count differs from crop count')
                    resolved.update(zip(requests, values))
                else:
                    for observed_fi, (_, _, _, _, valid) in zip(all_frames, all_items):
                        keys = [(observed_fi, tuple(h)) for _, h in valid if (observed_fi, tuple(h)) not in resolved]
                        flat = [requests[key] for key in keys]
                        started = time.perf_counter()
                        values = self.cls(flat) if flat else []
                        t_cls += time.perf_counter() - started
                        classifier_calls += bool(flat)
                        if len(values) != len(flat):
                            raise ValueError('Classifier output count differs from crop count')
                        resolved.update(zip(keys, values))
                predictions = [[resolved[(observed_fi, tuple(h))] for _, h in item[-1]]
                               for observed_fi, item in zip(all_frames, all_items)]
                pending_keys = {(sample[0], tuple(h)) for samples in pending.values() for sample in samples
                                for h, c in zip(sample[5], sample[6]) if c is not None}
                deferred_predictions = {key: resolved[key] for key in pending_keys if key in resolved}
                tentative_by_index = {idx: tid for tid, _, idx in tracker.tentative_measurements}
                for idx, item, values in zip(provisional_indices, provisional_items, predictions[len(track_items):]):
                    _, box, hbs, crops, valid = item
                    heads = [(h, p) for p, (_, h) in zip(values, valid)]
                    heads += [(h, None) for c, h in zip(crops, hbs) if c is None]
                    untracked_detections[fi].append({'detection_index': idx, 'box': box,
                        'motorcycle_score': ms[idx][1], 'heads': heads,
                        'candidate_track_id': tentative_by_index.get(idx), 'track_status': 'unconfirmed',
                        'source_index': timeline[fi]['source_index'], 'timestamp_s': timestamp})
                predictions = predictions[:len(track_items)]
                if trace:
                    trace({'frame': fi, 'stage': 'untracked_head_measurements', 'detections': untracked_detections[fi],
                           'valid_crops': sum(len(item[-1]) for item in provisional_items),
                           'crops': [{'motorcycle_index': idx, 'head_box': h, 'crop_box': square(h, margin=.15),
                                      'crop_shape': list(c.shape) if c is not None else None,
                                      'status': 'valid' if c is not None else 'too_small'}
                                     for idx, item in zip(provisional_indices, provisional_items)
                                     for h, c in zip(item[2], item[3])]})
            elif self.batch_heads:
                flat = [c for _, _, _, _, valid in track_items for c, _ in valid]
                t = time.perf_counter(); all_ps = self.cls(flat) if flat else []; t_cls += time.perf_counter() - t
                classifier_calls += bool(flat)
                if len(all_ps) != len(flat): raise ValueError("Classifier output count differs from crop count")
                offset = 0
                for _, _, _, _, valid in track_items:
                    predictions.append(all_ps[offset:offset+len(valid)]); offset += len(valid)
            else:
                for _, _, _, _, valid in track_items:
                    t = time.perf_counter(); ps = self.cls([c for c, _ in valid]) if valid else []; t_cls += time.perf_counter() - t
                    classifier_calls += bool(valid)
                    if len(ps) != len(valid): raise ValueError("Classifier output count differs from crop count")
                    predictions.append(ps)
            for observed_fi, (tid, tb, hbs, crops, valid), ps in zip(item_frames, track_items, predictions):
                ps_ok = [(p, c, h) for p, (c, h) in zip(ps, valid) if p is not None]
                all_probs = [p for p in ps] + [None for c in crops if c is None]
                vehicle_p = vehicle_probability(all_probs, self.hi, self.lo)
                if vehicle_p is not None:
                    pmax, cmax, hmax = max(ps_ok, key=lambda x: x[0])   # rider "tệ nhất" trên xe (lái + ngồi sau)
                    obs[tid].append({"frame": observed_fi, "source_index": timeline[observed_fi]["source_index"], "t": timeline[observed_fi]["timestamp_s"], "p_no_helmet": vehicle_p, "n_riders": len(hbs)})
                    key = (tid, pmax >= 0.5)
                    if evidence and (key not in best_ev or abs(pmax - 0.5) > abs(best_ev[key][0] - 0.5)):
                        best_ev[key] = (pmax, observed_fi, cmax.copy())
                frame_records[observed_fi].append((tid, tb, ps_ok[0][0] if ps_ok else None, [(h, p) for p, (c, h) in zip(ps, valid)] + [(h, None) for c_, h in zip(crops, hbs) if c_ is None]))
            for observed_fi in set(item_frames) | {fi}:
                frame_decisions[observed_fi] = {tid: decide([o["p_no_helmet"] for o in obs[tid] if o["frame"] <= observed_fi], self.n_min, self.hi, self.lo)[0]
                                                for tid, *_ in frame_records[observed_fi]}
            if trace:
                for observed_fi in sorted(set(item_frames) | {fi}):
                    trace({"frame": observed_fi, "processing_frame": fi, "stage": "classification", "heads": [
                        {"track_id": tid, "box": h, "probability": p, "label": decide([p], 1, self.hi, self.lo)[0]}
                        for tid, _, _, hbs in frame_records[observed_fi] for h, p in hbs]})
            if render or (out_dir and write_video):
                if out_dir and write_video and vw is None:
                    vw = cv2.VideoWriter(os.path.join(out_dir, f"{video_name}_annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (img.shape[1], img.shape[0]))
                vis = draw_annotations(img, frame_rec, frame_decisions[fi], timestamp, hi=self.hi, lo=self.lo,
                                       untracked=untracked_detections[fi])
                if vw is not None: vw.write(vis)
            frame_latency_ms.append((time.perf_counter() - frame_start) * 1000)
        if vw is not None: vw.release()
        discarded_pending += sum(len(samples) for samples in pending.values())
        self.last_vis = vis if (n and (render or (out_dir and write_video))) else None
        el = time.time() - t0
        results = {}; track_audit = {}
        for tid in {t for _, fr in per_frame for t, *_ in fr}:
            ps = [o["p_no_helmet"] for o in obs[tid]]
            lab, r, k = decide(ps, self.n_min, self.hi, self.lo)
            fr = [fi for fi, rec in per_frame if any(t == tid for t, *_ in rec)]
            st = track_stats(geo[tid])
            kept = not self.track_filter or keep_track(st, **self.track_filter)
            reason = "kept"
            if not kept:
                reason = ("f1_low_rider_evidence" if self.track_filter.get("tau_person") is not None
                          and st["person_score"] < self.track_filter["tau_person"] else "f2_low_displacement_low_rider_evidence")
            track_audit[tid] = {"track_id": tid, **st, **presence[tid], "track_detection_frames": len(fr),
                                "known_vehicle_observations": k, "unknown_observations": len(fr)-k,
                                "t_start": min(timeline[i]["timestamp_s"] for i in fr),
                                "t_end": max(timeline[i]["timestamp_s"] for i in fr),
                                "decision_before_filter": lab, "no_helmet_ratio": r, "n_riders": rider_counts[tid],
                                "kept": bool(kept), "reason": reason, "thresholds": self.track_filter}
            if trace: trace({"frame": None, "stage": "track_filter", "track_id": tid, "stats": st, "kept": kept})
            if not kept: continue
            results[tid] = {"track_id": tid, "decision": lab, "no_helmet_ratio": r, "n_obs": k,
                            "t_start": min(timeline[i]["timestamp_s"] for i in fr), "t_end": max(timeline[i]["timestamp_s"] for i in fr), "n_riders": rider_counts[tid], **st}
        if out_dir:
            with open(os.path.join(out_dir, f"{video_name}_tracks.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, ["track_id", "decision", "no_helmet_ratio", "n_obs", "t_start", "t_end", "n_riders", "person_score", "disp_norm", "n_frames", "evidence"]); w.writeheader()
                for tid, r in sorted(results.items()):
                    ev = ""
                    key = (tid, r["decision"] != OK)
                    if evidence and key in best_ev:
                        p, fi, c = best_ev[key]
                        ev = f"evidence/{video_name}_track{tid}_t{timeline[fi]['timestamp_s']:06.2f}s_p{p:.2f}.jpg"
                        cv2.imwrite(os.path.join(out_dir, ev), cv2.resize(c, (max(64, c.shape[1]*4), max(64, c.shape[0]*4)), interpolation=cv2.INTER_NEAREST))
                    w.writerow({**r, "evidence": ev})
            with open(os.path.join(out_dir, f"{video_name}_observations.csv"), "w", newline="") as f:
                w = csv.writer(f); w.writerow(["track_id", "frame", "t_sec", "p_no_helmet", "n_riders", "source_index", "inference_index"])
                for tid, os_ in obs.items():
                    for o in os_: w.writerow([tid, o["frame"], f"{o['t']:.6f}", f"{o['p_no_helmet']:.9f}", o["n_riders"], o["source_index"], o["frame"]])
            with open(os.path.join(out_dir, f"{video_name}_filter_audit.json"), "w", encoding="utf-8") as f:
                json.dump(track_audit, f, ensure_ascii=False, indent=2)
        return {"tracks": results, "per_frame": per_frame, "frames": n, "sec": el,
                "fps": n / el if el else 0, "t_det": t_det, "t_cls": t_cls,
                "classifier_calls": classifier_calls, "frame_latency_ms": frame_latency_ms,
                "timeline": timeline, "frame_decisions": frame_decisions, "observations": dict(obs),
                "untracked_detections": untracked_detections,
                "untracked_head_retention": {"enabled": self.retain_untracked_heads,
                    "heads": sum(len(d['heads']) for detections in untracked_detections.values() for d in detections),
                    "reused_predictions": reused_predictions, "policy": "measured_heads_without_confirmed_identity; no track votes"},
                "track_audit": track_audit, "tracking_audit": tracking_audit, "counters": counters,
                "temporal_backfill": {"enabled": bool(backfill), "vehicle_frames": len(backfill_events),
                    "discarded_vehicle_frames": discarded_pending, "events": backfill_events,
                    "skipped_without_valid_head_frames": skipped_headless,
                    "policy": "valid_measured_head_crops_only_after_identity_confirmation; no interpolation"},
                "f2": {"total_before": len(track_audit), "removed": len(track_audit)-len(results), "kept": len(results)}}
