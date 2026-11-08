"""Trace raw detections, threshold sweeps and rider ownership on static images.

The diagnostic classifier abstains: this tool measures head preservation, not
helmet classification or model latency. benchmark_crowded_riders.py runs the real
classifier separately. Baseline source is an explicit, preserved local snapshot.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import de14.pipeline as current
from de14.core import iou, crop, UNDET
from de14.metrics import evaluate_frames
from de14.runtime import load_profile


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_baseline(path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location('de14_debug_baseline', path / '__init__.py',
                                                 submodule_search_locations=[str(path)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    return __import__('de14_debug_baseline.pipeline', fromlist=['Pipeline'])


def ignored(box, regions):
    x, y = (box[0]+box[2])/2, (box[1]+box[3])/2
    return any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in regions)


def head_metrics(images, results):
    truth, predictions = [], []
    ignored_count = 0
    for image in images:
        key = dict(clip=image['id'], frame=0)
        truth.append({**key, 'vehicles': [dict(box=h['box'], violation=False, n_riders=1) for h in image['gt_heads']]})
        heads = []
        result = results[image['id']]
        for _, records in result['per_frame']:
            for tid, _, _, hs in records:
                if tid not in result['tracks']: continue
                for box, _ in hs:
                    if ignored(box, image.get('ignore_regions', [])):
                        ignored_count += 1
                    else: heads.append(dict(box=box, decision=UNDET, n_riders=1))
        predictions.append({**key, 'vehicles': heads})
    measured = evaluate_frames(truth, predictions, iou_threshold=.3)
    counts = measured['counts']; tp = counts['matched_vehicles']
    return {'scope': 'Associated heads after all filters, manually labeled development images; not helmet P/R or mAP',
            'iou_threshold': .3, 'gt': counts['gt_vehicles'], 'tp': tp,
            'fp': counts['predicted_vehicles']-tp, 'fn': counts['gt_vehicles']-tp,
            'precision': measured['vehicle_detection_precision'], 'recall': measured['vehicle_detection_recall'],
            'ignored_predictions': ignored_count}


def draw(image, entries, title, path):
    vis = image.copy()
    for box, text, color in entries:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        y = max(11, y1-3)
        cv2.putText(vis, text, (max(0, x1), y), cv2.FONT_HERSHEY_SIMPLEX, .36, (0, 0, 0), 3)
        cv2.putText(vis, text, (max(0, x1), y), cv2.FONT_HERSHEY_SIMPLEX, .36, color, 1)
    bar = np.zeros((25, vis.shape[1], 3), np.uint8)
    cv2.putText(bar, title, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, .44, (255, 255, 255), 1)
    cv2.imwrite(str(path), np.vstack([bar, vis]))


def score_for(box, detections):
    for h, s in zip(detections['head'], detections['head_score']):
        if iou(box, h) > .99: return s
    return None


def render_events(image, events, detections, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for step, event in enumerate(events):
        stage = event['stage']; entries = []
        if stage == 'detector_output':
            d = event['detections']
            for kind, color in [('motorcycle', (0, 220, 255)), ('person', (255, 150, 50)), ('head', (80, 255, 80))]:
                for index, (box, score) in enumerate(zip(d[kind], d.get(kind+'_score', [1]*len(d[kind])))):
                    entries.append((box, f'{kind[0]}{index} {score:.3f}', color))
        elif stage in {'vehicle_consolidation', 'motorcycle_confidence'}:
            keep = set(event.get('kept_indices', range(len(event['boxes']))))
            for index, (box, score) in enumerate(zip(event['boxes'], event['scores'])):
                entries.append((box, f'M{index} {score:.3f} '+('KEEP' if index in keep else 'REJECT'),
                                (80, 255, 80) if index in keep else (70, 70, 255)))
        elif stage == 'person_association':
            owners = {pi: mi for mi, people in event['owners'].items() for pi in people}
            for pi, box in enumerate(event['persons']):
                entries.append((box, f'P{pi}->M{owners.get(pi, "none")}', (255, 180, 50)))
        elif stage == 'head_association':
            decisions = event.get('decisions')
            if not decisions:
                decisions = [{'box': h, 'owner': mi} for mi, heads in event['owners'].items() for h in heads]
            for h in decisions:
                score = score_for(h['box'], detections)
                text = f'H {score:.3f}->M{h["owner"]}' if score is not None else f'H->M{h["owner"]}'
                entries.append((h['box'], text, (70, 255, 70) if h['owner'] is not None else (80, 80, 255)))
        elif stage == 'tracking':
            untracked = set(event['untracked_indices'])
            for index, (box, score) in enumerate(zip(event['input_boxes'], event['input_scores'])):
                entries.append((box, f'M{index} {score:.3f} '+('LOST' if index in untracked else 'RETAINED'),
                                (70, 70, 255) if index in untracked else (70, 255, 70)))
        elif stage in {'crops', 'classification'}:
            for index, h in enumerate(event['heads']):
                score = score_for(h['box'], detections)
                text = f'T{h["track_id"]} H{index}'+(f' {score:.3f}' if score is not None else '')
                entries.append((h.get('crop_box', h['box']), text+' '+h.get('status', 'independent crop'), (70, 255, 70)))
                if stage == 'crops':
                    patch_image = crop(image, h['crop_box'])
                    if patch_image is not None:
                        cv2.imwrite(str(folder/f'crop_T{h["track_id"]}_H{index}_{h["status"]}.png'), patch_image)
        else: continue
        draw(image, entries, stage, folder / f'{step:02d}_{stage}.png')
    (folder / 'trace.json').write_text(json.dumps(events, indent=2, ensure_ascii=False), encoding='utf-8')


def capture(images, cache, profile):
    """Use the same detector classes/checkpoints; save actual pre-NMS/pre-filter tensors."""
    from types import SimpleNamespace
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    from de14.detectors import YoloDetector, HeadEnhancedDetector
    from de14.headdet import DeimHeadDetector
    class Capture(DetectionPredictor):
        def postprocess(self, preds, img, orig_imgs, **kw):
            p = preds[0] if isinstance(preds, (list, tuple)) else preds
            self.debug_raw = p.detach().cpu().numpy().copy()
            self.debug_hw = list(img.shape[2:])
            self.debug_end2end = getattr(self.model, 'end2end', False)
            return super().postprocess(preds, img, orig_imgs, **kw)
    config = load_profile(profile)
    if config.device != 'cpu': raise ValueError('Capture currently supports the local CPU profiles')
    torch.set_num_threads(config.cpu_threads)
    yolo = YoloDetector(str(ROOT/config.yolo_weights), imgsz=config.imgsz, conf=.2)
    head = DeimHeadDetector(ROOT/config.head_path, device='cpu', min_score=.3, threads=config.ort_threads,
                            spinning=config.ort_spinning, backend=config.head_backend, openvino_device=config.openvino_device)
    raw = []
    if head.backend == 'onnx':
        session = head.s
        def run(*args, **kwargs):
            value = session.run(*args, **kwargs); raw.append(value[0].copy()); return value
        head.s = SimpleNamespace(run=run)
    else:
        request = head.request
        def infer(*args, **kwargs):
            value = request.infer(*args, **kwargs); raw.append(np.array(value[head._output], copy=True)); return value
        head.request = SimpleNamespace(infer=infer)
    for row in images:
        image = cv2.imread(str(ROOT/row['image']))
        yolo.m.predict(image, predictor=Capture, imgsz=config.imgsz, conf=.2, classes=[0, 3], device='cpu', verbose=False)
        raw.clear(); detections = HeadEnhancedDetector(yolo, head)(image)
        folder = cache/row['id']; folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder/'raw.npz', yolo=yolo.m.predictor.debug_raw, head=raw[0])
        record = {'image_sha256': row['image_sha256'], 'input_hw': yolo.m.predictor.debug_hw,
                  'end2end': yolo.m.predictor.debug_end2end, 'detector': detections}
        (folder/'detections.json').write_text(json.dumps(record, indent=2), encoding='utf-8')


def sweeps(image, record, tensors, folder, baseline):
    from ultralytics.utils import nms, ops
    folder.mkdir(parents=True, exist_ok=True)
    raw = torch.tensor(tensors['yolo']); height, width = image.shape[:2]
    results = []
    def replay(detections, **kwargs):
        pl = baseline.Pipeline(lambda _: detections, lambda cs: [None]*len(cs), fps=1, n_min=1, **kwargs)
        result = pl.run([(0, image)], write_video=False, evidence=False)
        return {'riders': sorted(t['n_riders'] for t in result['tracks'].values()),
                'heads': [{'track_id': tid, 'box': box, 'confidence': score_for(box, detections)}
                          for _, records in result['per_frame'] for tid, _, _, hs in records for box, _ in hs]}
    def final_evidence(name, result):
        draw(image, [(h['box'], f'T{h["track_id"]} H {h["confidence"]:.3f}', (70, 255, 70)) for h in result['heads']],
             name+' / legacy final heads', folder/f'{name}_legacy_final_heads.png')
        return result
    # Change exactly one parameter relative to conf=.2/iou=.7/class-aware/max_det=300.
    variants = [('baseline', .2, .7, False, 300)]
    variants += [(f'iou_{v}', .2, v, False, 300) for v in (.3, .5, .9)]
    variants += [(f'conf_{v}', v, .7, False, 300) for v in (.05, .1, .3)]
    variants += [('agnostic', .2, .7, True, 300)]
    variants += [(f'max_det_{v}', .2, .7, False, v) for v in (1, 3, 1000)]
    if not record['end2end']:
        scores, classes = raw[0, 4:].max(0)
        mask = ((classes == 0) | (classes == 3)) & (scores > .01)
        boxes = ops.xywh2xyxy(raw[0, :4].T[mask])
        boxes = ops.scale_boxes(record['input_hw'], boxes, image.shape)
        candidates = [{'box': b.tolist(), 'score': float(s), 'class': int(c)} for b, s, c in zip(boxes, scores[mask], classes[mask])]
        (folder/'yolo_pre_nms_candidates.json').write_text(json.dumps(candidates, indent=2), encoding='utf-8')
        shown = sorted(candidates, key=lambda r: -r['score'])[:80]
        draw(image, [(r['box'], f'C{r["class"]} {r["score"]:.3f}', (0, 200, 255)) for r in shown],
             'Pre-NMS: top 80 for drawing only; ALL raw rows in raw.npz', folder/'00_yolo_pre_nms.png')
    for name, conf, threshold, agnostic, maximum in variants:
        rows = nms.non_max_suppression(raw.clone(), conf, threshold, classes=[0, 3], agnostic=agnostic,
                                       max_det=maximum, end2end=record['end2end'])[0]
        rows[:, :4] = ops.scale_boxes(record['input_hw'], rows[:, :4], image.shape)
        entries = [(r[:4].tolist(), f'C{int(r[5])} {float(r[4]):.3f}', (80, 255, 80)) for r in rows]
        draw(image, entries, name, folder/f'yolo_{name}.png')
        data = {**record['detector'], 'person': rows[rows[:, 5] == 0, :4].tolist(),
                'person_score': rows[rows[:, 5] == 0, 4].tolist()}
        moto_rows = rows[rows[:, 5] == 3]; keep = []
        for index in range(len(moto_rows)):
            if not any(iou(moto_rows[index, :4], moto_rows[j, :4]) > .5 for j in keep): keep.append(index)
        data.update(motorcycle=moto_rows[keep, :4].tolist(), motorcycle_score=moto_rows[keep, 4].tolist())
        results.append({'name': name, 'confidence': conf, 'iou': threshold, 'agnostic': agnostic, 'max_det': maximum,
                        'persons': int((rows[:, 5] == 0).sum()), 'motorcycles': int((rows[:, 5] == 3).sum()), 'rows': rows.tolist()})
        final = final_evidence('yolo_'+name, replay(data))
        results[-1]['legacy_final_riders'] = final['riders']
        results[-1]['legacy_final_heads'] = final['heads']
        if name == 'baseline': default_motos = moto_rows
    rows = tensors['head'][0]; heads = rows[rows[:, 0].astype(int) == 7].copy()
    # The current exported model has normalized coordinates and no orig_target_sizes.
    heads[:, [1, 3]] *= width; heads[:, [2, 4]] *= height
    head_results = []
    for threshold in (.05, .1, .3, .5):
        selected = heads[heads[:, 5] >= threshold]
        draw(image, [(r[1:5], f'H {r[5]:.3f}', (80, 255, 80)) for r in selected],
             f'Head confidence >= {threshold}', folder/f'heads_conf_{threshold}.png')
        head_results.append({'threshold': threshold, 'rows_before_size_filter': selected.tolist(),
                             'kept_after_size_filter': int(sum((r[3]-r[1] >= 2) and (r[4]-r[2] >= 2) for r in selected))})
    head_nms = []
    for threshold in (.3, .5, .7, .9):
        kept = []
        for h in sorted(heads[heads[:, 5] >= .3], key=lambda r: -r[5]):
            if not any(iou(h[1:5], k[1:5]) > threshold for k in kept): kept.append(h)
        head_nms.append({'iou': threshold, 'kept': len(kept)})
    motorcycle_nms = []
    for threshold in (.3, .5, .7, .9):
        kept, rejected = [], []
        for index in range(len(default_motos)):
            suppressor = next((j for j in kept if iou(default_motos[index, :4], default_motos[j, :4]) > threshold), None)
            if suppressor is None: kept.append(index)
            else: rejected.append({'index': index, 'by': suppressor, 'iou': float(iou(default_motos[index, :4], default_motos[suppressor, :4]))})
        data = {**record['detector'], 'motorcycle': default_motos[kept, :4].tolist(), 'motorcycle_score': default_motos[kept, 4].tolist()}
        final = final_evidence(f'moto_nms_{threshold}', replay(data))
        motorcycle_nms.append({'iou': threshold, 'kept_indices': kept, 'rejected': rejected,
                               'legacy_final_riders': final['riders'], 'legacy_final_heads': final['heads']})
    moto_conf = []
    for value in (.2, .3, .4, .5):
        final = final_evidence(f'pipeline_conf_{value}', replay(record['detector'], conf_moto=value))
        moto_conf.append({'threshold': value, 'legacy_final_riders': final['riders'], 'legacy_final_heads': final['heads']})
    tracker_sweep = []
    original_tracker = baseline.ByteTrackWrapper
    for high, new in ((.4, .45), (.3, .45), (.4, .3), (.3, .3)):
        with patch.object(baseline, 'ByteTrackWrapper', lambda fps, **options: original_tracker(fps=fps, high=high, new=new, **options)):
            final = final_evidence(f'track_high_{high}_new_{new}', replay(record['detector']))
            tracker_sweep.append({'high': high, 'new': new, 'legacy_final_riders': final['riders'], 'legacy_final_heads': final['heads']})
    return {'actual_end2end': record['end2end'], 'raw_yolo_shape': list(raw.shape),
            'raw_head_shape': list(tensors['head'].shape), 'yolo': results,
            'head_confidence': head_results, 'head_nms': head_nms,
            'motorcycle_nms': motorcycle_nms, 'pipeline_motorcycle_confidence': moto_conf, 'tracker_thresholds': tracker_sweep}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--baseline-source', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--profile', default='openvino_cpu')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8')); images = manifest['images']
    for row in images:
        if digest(ROOT/row['image']) != row['image_sha256']: raise ValueError(f'Image hash mismatch: {row["id"]}')
    if any(not (args.cache/r['id']/'raw.npz').exists() for r in images): capture(images, args.cache, args.profile)
    baseline = load_baseline(args.baseline_source)
    torch.set_num_threads(4)
    modes = ('baseline', 'static_only', 'merge_only', 'association_only', 'combined_video', 'combined_static')
    all_results = {mode: {} for mode in modes}; summaries = {}
    for row in images:
        record = json.loads((args.cache/row['id']/'detections.json').read_text(encoding='utf-8'))
        if record['image_sha256'] != row['image_sha256']: raise ValueError('Stale detection cache')
        image = cv2.imread(str(ROOT/row['image'])); d = record['detector']; summaries[row['id']] = {}
        folder = args.out/row['id']; folder.mkdir(parents=True, exist_ok=True)
        draw(image, [(h['box'], h['vehicle'], (255, 220, 50)) for h in row['gt_heads']], 'Manual development GT', folder/'ground_truth.png')
        tensors = np.load(args.cache/row['id']/'raw.npz')
        sweep = sweeps(image, record, tensors, folder/'sweeps', baseline)
        (folder/'sweeps.json').write_text(json.dumps(sweep, indent=2), encoding='utf-8')
        for mode in modes:
            events = []; module = baseline if mode == 'baseline' else current
            pipeline = module.Pipeline(lambda _: d, lambda cs: [None]*len(cs), fps=1, n_min=1, batch_heads=True)
            identity = lambda boxes, scores, persons, ps, *a: (boxes, scores, [[i] for i in range(len(boxes))])
            old_heads = lambda motos, hs, **kw: baseline.heads_for_motorcycles(motos, hs)
            with patch.object(current, 'merge_motorcycle_fragments', identity if mode in {'static_only', 'association_only'} else current.merge_motorcycle_fragments):
                with patch.object(current, 'heads_for_motorcycles', old_heads if mode in {'static_only', 'merge_only'} else current.heads_for_motorcycles):
                    options = {} if mode == 'baseline' else {'tracking': mode not in {'static_only', 'combined_static'}}
                    result = pipeline.run([(0, image)], write_video=False, evidence=False, trace=events.append, **options)
            all_results[mode][row['id']] = result
            summaries[row['id']][mode] = {'riders': sorted(t['n_riders'] for t in result['tracks'].values()),
                                           'heads': sum(t['n_riders'] for t in result['tracks'].values())}
            render_events(image, events, d, folder/mode)
            entries = [(box, f'T{tid} H {score_for(box,d):.3f}', (70, 255, 70))
                       for _, fr in result['per_frame'] for tid, _, _, hs in fr for box, _ in hs]
            draw(image, entries, mode, folder/f'{mode}.png')
        before = cv2.imread(str(folder/'baseline.png')); after = cv2.imread(str(folder/'combined_static.png'))
        cv2.imwrite(str(folder/'comparison.png'), np.hstack([before, after]))
        print(row['id'], json.dumps(summaries[row['id']]), flush=True)
    report = {'manifest_sha256': digest(args.manifest), 'scope': manifest['scope'], 'images': summaries,
              'metrics': {mode: head_metrics(images, result) for mode, result in all_results.items()},
              'baseline_source_sha256': {p.name: digest(p) for p in args.baseline_source.glob('*.py')},
              'current_source_sha256': {name: digest(ROOT/'src/de14'/name) for name in ('pipeline.py','core.py','detectors.py','headdet.py')},
              'limits': ['Cached detection replay has no valid end-to-end timing.', 'Diagnostic classifier always abstains.',
                         'P/R measures head preservation, not helmet classification or correct vehicle ownership.',
                         'Six development images, coarse manual labels by one reviewer, not independent test accuracy.']}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report['metrics'], indent=2), flush=True)


if __name__ == '__main__': main()
