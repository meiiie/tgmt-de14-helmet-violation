"""Compare legacy estimated regions, localized heads and optional CLIP on the same models.

All inference is real; only weights/instances are shared. This does not compare
standalone memory footprints because the four-model control stays resident.
"""
import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from de14.runtime import build_runtime, load_profile, RuntimeBundle
from de14.core import iou, decide, VIOL, OK, UNDET
from de14.metrics import evaluate_frames
from debug_rider_pipeline import digest, draw, head_metrics
from benchmark_crowded_riders import vehicle_predictions, timing
from benchmark_runtime import VramSampler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--official-manifest', type=Path)
    parser.add_argument('--profile', default='openvino_cpu')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1: parser.error('repeats must be positive')
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    images = {}
    for row in manifest['images']:
        path = ROOT/row['image']
        if digest(path) != row['image_sha256']: raise ValueError(f'Input hash mismatch: {path}')
        images[row['id']] = cv2.imread(str(path))
    config = load_profile(args.profile, accurate=True, detect_heads=True, verify_with_clip=True)
    if config.detector_kind != 'yolo': raise ValueError('Legacy estimated-head control requires the YOLO detector')
    start = time.perf_counter(); strict = build_runtime(config); load_seconds = time.perf_counter()-start
    start = time.perf_counter(); tight = build_runtime(replace(config, accurate=False), reuse=strict)
    switch_seconds = time.perf_counter()-start
    # The legacy path used exactly this YOLO detector and bare probe; use their
    # existing instances to hold hardware/weights fixed without duplicated loads.
    legacy = RuntimeBundle(strict.detector.det, strict.classifier.base,
                           replace(config, accurate=False, detect_heads=False), strict.device)
    modes = {'legacy_estimated': legacy, 'tight_dino': tight, 'tight_clip': strict}
    sharing = {'detector_shared': tight.detector is strict.detector,
               'base_classifier_shared': tight.classifier.base is strict.classifier.base,
               'legacy_yolo_shared': legacy.detector is strict.detector.det}
    assert all(sharing.values()), sharing
    cuda = strict.device.startswith('cuda')
    def sync():
        if cuda: torch.cuda.synchronize(strict.device)
    def run(name, frames, static=True):
        sync()
        result = modes[name].pipeline(fps=1 if static else 10, n_min=1).run(
            frames, write_video=False, evidence=False, tracking=not static)
        sync()
        return result
    for name in modes: run(name, [(0, next(iter(images.values())))])
    if cuda: torch.cuda.reset_peak_memory_stats(strict.device)
    vram = VramSampler(cuda)
    samples = {m: [] for m in modes}; results = {m: {} for m in modes}
    per_image = {m: defaultdict(list) for m in modes}; stages = {m: {'det':0., 'cls':0.} for m in modes}
    for repeat in range(args.repeats):
        order = list(modes) if repeat % 2 == 0 else list(reversed(modes))
        for name in order:
            for image_id, image in images.items():
                result = run(name, [(0, image)])
                samples[name].extend(result['frame_latency_ms'])
                per_image[name][image_id].extend(result['frame_latency_ms'])
                stages[name]['det'] += result['t_det']; stages[name]['cls'] += result['t_cls']
                if repeat == 0: results[name][image_id] = result
                print('image', repeat+1, name, image_id, result['frame_latency_ms'],
                      [t['n_riders'] for t in result['tracks'].values()], flush=True)
            # Preserve completed inference if a later metric/export fails. These
            # are explicitly partial artifacts, never a completed report.
            (args.out/'image_progress.json').write_text(json.dumps(
                {'complete': False, 'repeat': repeat+1, 'last_mode': name,
                 'samples': samples, 'results': results}, ensure_ascii=False), encoding='utf-8')
    official = None
    if args.official_manifest:
        gt = json.loads(args.official_manifest.read_text(encoding='utf-8')); groups = defaultdict(list)
        for row in gt['frames']:
            path = ROOT/row['image']
            if digest(path) != row['sha256']: raise ValueError('Official input hash mismatch')
            image = cv2.resize(cv2.imread(str(path)), tuple(gt['source_resolution']), interpolation=cv2.INTER_CUBIC)
            groups[row['clip']].append((row['frame'], image))
        official_samples = {m: [] for m in modes}; predictions = {m: [] for m in modes}
        for name in modes:
            for clip, frames in sorted(groups.items()):
                result = run(name, sorted(frames), static=False)
                official_samples[name].extend(result['frame_latency_ms'])
                predictions[name].extend(vehicle_predictions(clip, result))
                print('official', name, clip, result['frame_latency_ms'], flush=True)
                (args.out/'official_progress.json').write_text(json.dumps(
                    {'complete': False, 'last_mode': name, 'last_clip': clip,
                     'samples': official_samples, 'predictions': predictions}, ensure_ascii=False), encoding='utf-8')
        official = {'manifest_sha256': digest(args.official_manifest), 'repeats': 1,
                    'scope': '16 correlated official-test frames; vehicle/frame metrics, one timing repeat',
                    'modes': {m: {'latency': timing(official_samples[m]), 'metrics': evaluate_frames(gt['frames'], predictions[m])} for m in modes}}
    vram.close()
    summaries = {}
    for name, bundle in modes.items():
        summaries[name] = {'runtime': bundle.metadata(), 'latency': timing(samples[name]),
                           'head_metrics': head_metrics(manifest['images'], results[name]),
                           'stage_mean_ms': {k: v*1000/len(samples[name]) for k, v in stages[name].items()}, 'images': {}}
        for image_id, result in results[name].items():
            heads = [{'box': h, 'p_no_helmet': p, 'label': decide([p],1)[0]}
                     for _, records in result['per_frame'] for _, _, _, hs in records for h,p in hs]
            summaries[name]['images'][image_id] = {'riders': sorted(t['n_riders'] for t in result['tracks'].values()),
                                                  'heads': heads, 'latency': timing(per_image[name][image_id])}
            entries = []
            for _, records in result['per_frame']:
                for tid, box, _, hs in records:
                    entries.append((box, f'#{tid} riders={len(hs)}', (0,210,255)))
                    for h,p in hs:
                        label = decide([p],1)[0]
                        text = 'NO HELMET' if label == VIOL else 'HELMET' if label == OK else 'UNKNOWN'
                        color = (50,50,255) if label == VIOL else (60,255,60) if label == OK else (0,220,255)
                        entries.append((h,text,color))
            draw(images[image_id], entries, name, args.out/f'{image_id}_{name}.png')
    for image_id in images:
        panels = [cv2.imread(str(args.out/f'{image_id}_{name}.png')) for name in modes]
        cv2.imwrite(str(args.out/f'{image_id}_comparison.png'), np.hstack(panels))
    report = {'profile': args.profile, 'manifest_sha256': digest(args.manifest), 'repeats': args.repeats,
              'initial_four_model_loading_seconds': load_seconds, 'switch_to_tight_dino_seconds': switch_seconds,
              'shared_instances': sharing, 'modes': summaries, 'official': official,
              'vram': {'process_peak_mib': vram.peak, 'note': vram.note,
                       'scope': 'Shared four-model resident benchmark; not standalone mode memory comparison'},
              'checkpoint_sha256': {key: digest(ROOT/path) for key,path in {'yolo':config.yolo_weights,'head':config.head_path,
                  'probe':config.probe_path,'clip':str(Path(config.clip_path)/'model.safetensors')}.items()},
              'source_sha256': {name:digest(ROOT/name) for name in ('src/de14/runtime.py','src/de14/pipeline.py','src/de14/classifiers.py','app.py','scripts/benchmark_head_modes.py')},
              'limits': ['Development image labels are coarse manual rider-head boxes, not helmet GT or independent test.',
                         'Compute latency excludes decode/load/export; all inference is rerun.',
                         'Video timing is one short-window repeat and may drift; keep raw results.',
                         'No thresholds/weights were fitted to these images; CLIP is optional and can abstain on real helmets.']}
    (args.out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({m:{k:summaries[m][k] for k in ('latency','head_metrics','stage_mean_ms')} for m in modes},indent=2),flush=True)


if __name__ == '__main__': main()
