"""Frozen-model temporal benchmark, shared across local CPU and cloud CUDA.

Three uncached inference runs (stride 2/5/10), each replayed at n_min 5/10.
No labels are inferred from confidence: GT-dependent metrics stay null without GT.
"""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from de14.core import decide, UNDET, VIOL
from de14.video_io import run_video, render_video, json_default


def sha256(path):
    with open(path, 'rb') as file: return hashlib.file_digest(file, 'sha256').hexdigest()


def save(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(data, default=json_default, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def encoder_identity(encoder):
    """Fingerprint the actual loaded tensors and preprocessing, including local overrides."""
    import torch
    checksum = hashlib.sha256()
    for name, tensor in sorted(encoder.model.state_dict().items()):
        checksum.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        checksum.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return {'model_id': encoder.model_id, 'loaded_state_sha256': checksum.hexdigest(),
            'processor': encoder.proc.to_dict(), 'processor_class': type(encoder.proc).__name__}


def validate_provenance(path, provenance, resume=False):
    """Reject mismatches before touching the existing evidence manifest."""
    path = Path(path)
    if path.exists():
        old = json.loads(path.read_text(encoding='utf-8'))
        if not resume or old.get('key') != provenance['key']:
            raise ValueError('Existing provenance differs or resume was not requested; use a new output folder')
    else:
        save(path, provenance)


def replay_votes(result, n_min):
    """Replay unrounded known votes; classification/tracking/F2 remain identical."""
    replay = copy.deepcopy(result)
    for tid, audit in replay['track_audit'].items():
        observations = replay['observations'].get(tid, [])
        lab, ratio, count = decide([o['p_no_helmet'] for o in observations], n_min,
                                   replay.get('hi', .6), replay.get('lo', .4))
        audit.update(decision_before_filter=lab, no_helmet_ratio=ratio)
        if tid in replay['tracks']:
            replay['tracks'][tid].update(decision=lab, no_helmet_ratio=ratio, n_obs=count)
    replay['frame_decisions'] = {}
    for fi, records in replay['per_frame']:
        replay['frame_decisions'][fi] = {
            tid: decide([o['p_no_helmet'] for o in replay['observations'].get(tid, []) if o['frame'] <= fi],
                        n_min, replay.get('hi', .6), replay.get('lo', .4))[0] for tid, *_ in records}
    return replay


def summarize(result, stride, n_min):
    tracks = result['tracks']; unknown = sum(t['decision'] == UNDET for t in tracks.values())
    all_tracks = list(result['track_audit'].values())
    unknown_shadow = sum(t['decision_before_filter'] == UNDET for t in all_tracks)
    return {'stride': stride, 'min_observations': n_min, 'sampled_frames': result['frames'],
            'sample_interval_s': result.get('media', {}).get('sample_interval_s'),
            'vehicle_tracks': len(tracks), 'person_detections': result['counters']['person_detections'],
            'counters': result['counters'], 'unknown_tracks': unknown,
            'unknown_rate': unknown/len(tracks) if tracks else None, 'unknown_denominator': len(tracks),
            'violation_tracks': sum(t['decision'] == VIOL for t in tracks.values()),
            'f2': result['f2'], 'f2_shadow_off': {'vehicle_tracks': len(all_tracks),
                'unknown_rate': unknown_shadow/len(all_tracks) if all_tracks else None,
                'violation_tracks': sum(t['decision_before_filter'] == VIOL for t in all_tracks)},
            'tracks_with_fewer_known_observations_than_min': sum(t['known_vehicle_observations'] < n_min for t in all_tracks),
            'precision': None, 'recall': None, 'fp': None, 'fn': None,
            'id_switches': None, 'gt_fragmentation': None, 'quality_status': 'N/A: verified temporal GT is not available',
            'processing_seconds': result['sec'],
            'processing_scope': 'Sample decode + inference/tracking loop; excludes model load, source PTS probe and render/export',
            'source_probe_seconds': result.get('media', {}).get('probe_seconds'),
            'first_full_run_wall_seconds': result.get('total_seconds'),
            'sampled_processing_fps': result['frames']/result['sec'] if result['sec'] else None,
            'model_inference_seconds': result['t_det']+result['t_cls'],
            'model_inference_fps': result['frames']/(result['t_det']+result['t_cls']) if result['t_det']+result['t_cls'] else None,
            'detector_seconds': result['t_det'], 'classifier_seconds': result['t_cls'],
            'output_duration': result.get('media', {}).get('output', {}).get('video_duration'),
            'output_fps': result.get('media', {}).get('output', {}).get('fps'),
            'output_frames': result.get('media', {}).get('written_frames'),
            'render_encode_seconds': result.get('media', {}).get('render_encode_seconds'),
            'tracker_buffer_seconds': result['tracking_audit'][0]['lost_buffer_seconds'] if result['tracking_audit'] else None,
            'measurement_note': 'Uncached inference, shared within same-stride n_min pair; n_min replay/render are timed separately'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--out', type=Path, default=ROOT/'outputs/benchmarks/video_sampling_20261106')
    parser.add_argument('--profile', default='openvino_cpu')
    parser.add_argument('--imgsz', type=int, default=1280)
    parser.add_argument('--clip', action='store_true', help='Frozen across all strides; default off per latest user preference')
    parser.add_argument('--strides', type=int, nargs='+', default=[10, 5, 2])
    parser.add_argument('--resume', action='store_true', help='Reuse completed identical-config stride runs, never report replay as live inference')
    args = parser.parse_args()
    from de14.runtime import load_profile, build_runtime
    from benchmark_runtime import VramSampler
    import torch
    filtering = json.loads((ROOT/'configs/track_filter.json').read_text())['track_filter']
    config = load_profile(args.profile, imgsz=args.imgsz, accurate=args.clip, detect_heads=True)
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter(); runtime = build_runtime(config)
    load_seconds = time.perf_counter()-start
    base = getattr(runtime.classifier, 'base', runtime.classifier)
    weights = [ROOT/config.yolo_weights, ROOT/config.head_path, ROOT/config.probe_path]
    weights.append(ROOT/'weights/clip-vit-base-patch32/model.safetensors') if args.clip else None
    sources = list((ROOT/'src/de14').glob('*.py'))+[Path(__file__)]
    packages = {}
    for name in ('torch', 'transformers', 'ultralytics', 'numpy', 'openvino', 'onnxruntime',
                 'onnxruntime-gpu', 'opencv-python', 'opencv-python-headless', 'scikit-learn'):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    provenance = {'source_sha256': sha256(args.source), 'config': asdict(config), 'filter': filtering,
                  'weights_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in weights},
                  'code_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in sources},
                  'hardware': platform.platform(), 'profile': args.profile,
                  'encoder_identity': encoder_identity(base.e),
                  'packages': packages,
                  'ground_truth': None, 'comparison': 'new frozen configuration; historical 366.9s config was not captured',
                  'cache_policy': 'No detector/classifier prediction cache; only n_min voting reuses measurements'}
    provenance_key = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    validate_provenance(args.out/'provenance.json', {**provenance, 'key': provenance_key}, resume=args.resume)
    memory = VramSampler(runtime.device.startswith('cuda'))
    rows = []; runs = {}
    try:
        for stride in args.strides:
            folder = args.out/f'stride{stride}'
            snapshot = folder/'benchmark_measurements.json'
            result = None
            if args.resume and snapshot.exists():
                saved = json.loads(snapshot.read_text(encoding='utf-8'))
                if saved['provenance_key'] != provenance_key: raise ValueError('Resume provenance mismatch')
                result = saved['result']
                for name in ('tracks', 'observations', 'track_audit', 'frame_decisions', 'timeline'):
                    result[name] = {int(k): v for k, v in result[name].items()}
                result['frame_decisions'] = {fi: {int(t): lab for t, lab in d.items()} for fi, d in result['frame_decisions'].items()}
                print(f'Resumed completed stride {stride} with saved live timings', flush=True)
            if result is None:
                pipeline = runtime.pipeline(n_min=5, track_filter=filtering)
                # Progress is emitted from actual NN calls; no synthetic tracker updates.
                original_detector = pipeline.det
                count = [0]; began = time.perf_counter()
                def detector(image):
                    output = original_detector(image); count[0] += 1
                    if count[0] == 1 or count[0] % 10 == 0:
                        print(f'stride={stride} sampled={count[0]} elapsed={time.perf_counter()-began:.1f}s', flush=True)
                    return output
                pipeline.det = detector
                result = run_video(args.source, pipeline, folder, 'video', stride=stride)
                save(snapshot, {'provenance_key': provenance_key, 'result': result})
            runs[stride] = {'frames': result['frames'], 'live_processing_seconds': result['sec']}
            for minimum in (5, 10):
                replay_start = time.perf_counter(); replay = replay_votes(result, minimum)
                replay_seconds = time.perf_counter()-replay_start
                destination = args.out/f'S{stride}_M{minimum}'
                destination.mkdir(parents=True, exist_ok=True)
                replay['video_path'] = render_video(args.source, replay['media']['source'], replay,
                                                   destination/'full_length_annotated.mp4')
                save(destination/'predictions.json', replay)
                with (destination/'tracks.csv').open('w', newline='', encoding='utf-8') as file:
                    fieldnames = ['track_id', 'decision', 'n_obs', 'n_riders', 't_start', 't_end', 'no_helmet_ratio']
                    writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction='ignore'); writer.writeheader()
                    writer.writerows(replay['tracks'].values())
                row = summarize(replay, stride, minimum)
                row.update(vote_replay_seconds=replay_seconds, video_path=replay['video_path'])
                rows.append(row)
                save(args.out/'progress.json', {'status': 'partial', 'rows': rows, 'runs': runs})
                print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        memory.close()
    report = {'status': 'complete_operational_benchmark_quality_pending_gt', 'provenance': provenance,
              'runtime': runtime.metadata(), 'model_load_seconds': load_seconds, 'rows': rows,
              'torch_peak_vram_mb': torch.cuda.max_memory_allocated(runtime.device)/2**20 if runtime.device.startswith('cuda') else None,
              'process_peak_vram_mb': memory.peak, 'vram_note': memory.note}
    save(args.out/'report.json', report)
    with (args.out/'report.csv').open('w', newline='', encoding='utf-8') as file:
        fields = ['stride', 'min_observations', 'sampled_frames', 'vehicle_tracks', 'person_detections',
                  'violation_tracks', 'unknown_rate', 'precision', 'recall', 'fp', 'fn', 'id_switches',
                  'gt_fragmentation', 'processing_seconds', 'model_inference_fps', 'output_duration', 'output_fps', 'output_frames']
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)


if __name__ == '__main__': main()
