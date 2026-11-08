"""Paired real-model benchmark of preserved/current pipelines on identical inputs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from de14.runtime import load_profile, build_runtime
from de14.core import decide, VIOL, OK, UNDET
from de14.metrics import evaluate_frames
from debug_rider_pipeline import load_baseline, head_metrics, digest, draw


def vehicle_predictions(clip, result):
    output = []
    for frame, records in result['per_frame']:
        vehicles = []
        for _, box, _, heads in records:
            labels = [decide([p], 1)[0] for _, p in heads]
            decision = VIOL if VIOL in labels else OK if labels and all(l == OK for l in labels) else UNDET
            vehicles.append({'box': box, 'decision': decision, 'n_riders': len(heads),
                             'heads': [{'box': h, 'p_no_helmet': p, 'label': label} for (h, p), label in zip(heads, labels)]})
        output.append({'clip': clip, 'frame': frame, 'vehicles': vehicles})
    return output


def timing(values):
    return {'frames': len(values), 'p50_ms': float(np.percentile(values, 50)),
            'p95_ms': float(np.percentile(values, 95)), 'fps_compute': len(values)/(sum(values)/1000)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--official-manifest', type=Path)
    parser.add_argument('--baseline-source', type=Path, required=True)
    parser.add_argument('--profile', default='openvino_cpu')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--official-repeats', type=int, default=1)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.official_repeats < 1: parser.error('Repeats must be positive')
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    images = {}
    for row in manifest['images']:
        path = ROOT/row['image']
        if digest(path) != row['image_sha256']: raise ValueError(f'Image hash mismatch: {path}')
        images[row['id']] = cv2.imread(str(path))
    baseline = load_baseline(args.baseline_source)
    config = load_profile(args.profile)
    start = time.perf_counter(); bundle = build_runtime(config); load_seconds = time.perf_counter()-start
    cuda = bundle.device.startswith('cuda')
    def sync():
        if cuda: torch.cuda.synchronize(bundle.device)
    def pipeline(mode, fps):
        return (baseline.Pipeline(bundle.detector, bundle.classifier, fps=fps, n_min=1, batch_heads=config.batch_heads)
                if mode == 'before' else bundle.pipeline(fps=fps, n_min=1))
    def run(mode, frames, static=False, fps=1):
        options = {'tracking': not static} if mode == 'after' else {}
        sync()
        result = pipeline(mode, fps).run(frames, write_video=False, evidence=False, **options)
        sync()
        return result
    args.out.mkdir(parents=True, exist_ok=True)
    warm_start = time.perf_counter()
    for mode in ('before', 'after'):
        run(mode, [(0, next(iter(images.values())))], static=True)
    warm_seconds = time.perf_counter()-warm_start
    if cuda: torch.cuda.reset_peak_memory_stats(bundle.device)
    samples = {m: [] for m in ('before', 'after')}; results = {m: {} for m in samples}
    per_image = {m: defaultdict(list) for m in samples}
    for repeat in range(args.repeats):
        for mode in (('before', 'after') if repeat % 2 == 0 else ('after', 'before')):
            for name, image in images.items():
                result = run(mode, [(0, image)], static=True)
                samples[mode].extend(result['frame_latency_ms']); per_image[mode][name].extend(result['frame_latency_ms'])
                if repeat == 0: results[mode][name] = result
                print('crowded', repeat+1, mode, name, result['frame_latency_ms'],
                      [r['n_riders'] for r in result['tracks'].values()], flush=True)
    official = None
    if args.official_manifest:
        official_manifest = json.loads(args.official_manifest.read_text(encoding='utf-8'))
        frames = defaultdict(list)
        for row in official_manifest['frames']:
            path = ROOT/row['image']
            if digest(path) != row['sha256']: raise ValueError('Official frame hash mismatch')
            image = cv2.imread(str(path))
            image = cv2.resize(image, tuple(official_manifest['source_resolution']), interpolation=cv2.INTER_CUBIC)
            frames[row['clip']].append((row['frame'], image))
        official_predictions = {m: [] for m in samples}; official_samples = {m: [] for m in samples}
        for repeat in range(args.official_repeats):
            for mode in (('before', 'after') if repeat % 2 == 0 else ('after', 'before')):
                for clip, rows in sorted(frames.items()):
                    result = run(mode, sorted(rows, key=lambda r:r[0]), fps=10)
                    official_samples[mode].extend(result['frame_latency_ms'])
                    if repeat == 0: official_predictions[mode].extend(vehicle_predictions(clip, result))
                    print('official', repeat+1, mode, clip, 'ms', result['frame_latency_ms'], flush=True)
        official = {'manifest_sha256': digest(args.official_manifest), 'repeats': args.official_repeats,
                    'scope': 'Same 16 correlated official-test frames as prior report; vehicle/frame metrics, short tracking windows',
                    'profiles': {m: {'latency': timing(official_samples[m]),
                                     'metrics': evaluate_frames(official_manifest['frames'], official_predictions[m])} for m in samples}}
        (args.out/'official_predictions.json').write_text(json.dumps(official_predictions, indent=2, ensure_ascii=False), encoding='utf-8')
    summaries = {}
    for mode in samples:
        summaries[mode] = {'latency': timing(samples[mode]), 'head_metrics': head_metrics(manifest['images'], results[mode]), 'images': {}}
        for name, result in results[mode].items():
            summaries[mode]['images'][name] = {'latency': timing(per_image[mode][name]),
                'riders': sorted(t['n_riders'] for t in result['tracks'].values()),
                'heads': [{'box': box, 'probability': p, 'label': decide([p], 1)[0]}
                          for _, records in result['per_frame'] for _, _, _, hs in records for box, p in hs]}
            entries = []
            for _, records in result['per_frame']:
                for tid, vehicle_box, _, heads in records:
                    entries.append((vehicle_box, f'#{tid} riders={len(heads)}', (0, 210, 255)))
                    for box, p in heads:
                        label = decide([p], 1)[0]
                        text = 'NO HELMET' if label == VIOL else 'HELMET' if label == OK else 'UNKNOWN'
                        color = (50, 50, 255) if label == VIOL else (60, 255, 60) if label == OK else (0, 220, 255)
                        entries.append((box, text+(f' p={p:.2f}' if p is not None else ''), color))
            draw(images[name], entries, mode+' / real classifier', args.out/f'{name}_{mode}.png')
    report = {'profile': args.profile, 'runtime': bundle.metadata(), 'model_loading_seconds': load_seconds,
              'warmup_seconds': warm_seconds, 'repeats': args.repeats, 'manifest_sha256': digest(args.manifest),
              'checkpoint_sha256': {k: digest(ROOT/p) for k, p in {'yolo': config.yolo_weights, 'head': config.head_path,
                  'probe': config.probe_path, 'clip': str(Path(config.clip_path)/'model.safetensors')}.items()},
              'source_sha256': {'before': {p.name:digest(p) for p in args.baseline_source.glob('*.py')},
                               'after': {p.name:digest(p) for p in (ROOT/'src/de14').glob('*.py')}},
              'crowded': summaries, 'official': official,
              'vram': {'torch_peak_mib': torch.cuda.max_memory_allocated(bundle.device)/2**20 if cuda else None,
                       'note': 'Torch-only peak excludes ORT GPU allocations; CPU/Intel shared memory is not measured'},
              'limits': ['Compute timing excludes decode/load/export; model inference is NOT cached.',
                         'Same loaded models and inputs for both pipelines; static images bypass temporal gates only after fix.',
                         'Six hand-labeled development images are not held-out accuracy; head P/R is not helmet P/R.',
                         'Official held-out frame metrics are separate; one short-window repeat has high uncertainty.']}
    (args.out/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'crowded': {m: {k:summaries[m][k] for k in ('latency','head_metrics')} for m in samples},
                      'official': official}, indent=2, ensure_ascii=True), flush=True)


if __name__ == '__main__': main()
