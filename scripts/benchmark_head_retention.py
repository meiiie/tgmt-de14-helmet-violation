"""Paired frozen-detector replay with real classification of newly retained crops."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from de14.classifiers import DinoProbe, VerifiedHelmetCls
from de14.core import crop, decide, UNDET
from de14.headdet import square
from de14.pipeline import Pipeline
from de14.video_io import sampled_frames, json_default
from benchmark_video_tracking import CropCache, supplied_box_coverage, matches


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, default=json_default, ensure_ascii=False, indent=2), encoding='utf8')


def benchmark(source, baseline_folder, output, device='cpu', labels_path=None):
    import torch
    torch.set_num_threads(4)
    baseline_folder = Path(baseline_folder)
    output = Path(output)
    old = json.loads((baseline_folder / 'video_audit.json').read_text(encoding='utf8'))
    report = json.loads((baseline_folder / 'report.json').read_text(encoding='utf8'))
    from benchmark_video_sampling import sha256, encoder_identity
    if sha256(source) != report['experiment']['source_sha256']:
        raise ValueError('Source video differs from baseline')
    for relative, expected in report['experiment']['weights'].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError('Model weight differs from baseline')
    base = DinoProbe.load(str(ROOT / 'weights/dino_probe.joblib'), device=device, local_files_only=True)
    encoder = encoder_identity(base.e)
    if encoder['loaded_state_sha256'] != report['experiment']['encoder']['loaded_state_sha256']:
        raise ValueError('Encoder tensor identity differs from baseline')
    classifier = CropCache(VerifiedHelmetCls(base, None))
    info = old['media']['source']
    stride = old['media']['stride']
    raw = {}
    with (baseline_folder / 'video_stages.jsonl').open(encoding='utf8') as file:
        for line in file:
            event = json.loads(line)
            if event['stage'] == 'detector_output':
                raw[event['frame']] = event['detections']
    old_records = {old['timeline'][str(fi)]['source_index']: records for fi, records in old['per_frame']}
    timeline = {}
    for fi, image in sampled_frames(source, info, stride, timeline=timeline):
        for _, _, _, heads in old_records[timeline[fi]['source_index']]:
            for head, probability in heads:
                pixels = crop(image, square(head, margin=.15), 8)
                if pixels is not None:
                    classifier.values[CropCache.key(pixels)] = probability
    labels_path = labels_path or ROOT / 'outputs/label_review_20261106/labels_received.json'
    labels = json.loads(Path(labels_path).read_text(encoding='utf8'))
    if labels['source']['sha256'] != report['experiment']['source_sha256']:
        raise ValueError('Annotation source differs from video')
    sampled = {meta['source_index'] for meta in old['timeline'].values()}
    selected = [frame for frame in labels['frames'] if frame['reviewed'] and frame['source_index'] in sampled]
    rows = []
    for name, retain in [('before', False), ('after', True)]:
        sequence = iter(copy.deepcopy(list(raw.values())))
        timeline = {}
        before_misses = classifier.misses
        before_forward = classifier.forward_seconds
        pipeline = Pipeline(lambda _: next(sequence), classifier, fps=info['fps'] / stride,
                            n_min=report['experiment']['n_min'], batch_heads=True,
                            track_filter=report['experiment']['filter'], retain_untracked_heads=retain)
        started = time.perf_counter()
        result = pipeline.run(sampled_frames(source, info, stride, timeline=timeline),
                              write_video=False, evidence=False, frame_metadata=timeline)
        if name == 'before':
            if json.dumps(result['per_frame'], default=json_default) != json.dumps(old['per_frame']):
                raise AssertionError('Baseline per-frame replay differs')
            if result['tracks'] != {int(k): v for k, v in old['tracks'].items()}:
                raise AssertionError('Baseline track/vote replay differs')
        confirmed = supplied_box_coverage(result, {**labels, 'frames': selected})
        measured = 0
        for frame in selected:
            fi = frame['source_index'] // stride
            boxes = [head['box'] for vehicle in frame['vehicles'] for head in vehicle['heads']]
            records = result['per_frame'][fi][1]
            predicted = [head for tid, _, _, heads in records if tid in result['tracks'] for head, _ in heads]
            predicted += [head for detection in result['untracked_detections'][fi] for head, _ in detection['heads']]
            measured += matches(boxes, predicted, .5)
        row = {'variant': name, 'confirmed_box_coverage': confirmed,
               'all_measured_head_box_matches': measured, 'provided_head_boxes': sum(len(v['heads']) for f in selected for v in f['vehicles']),
               'reviewed_analyzed_frames': len(selected), 'vehicle_tracks': len(result['tracks']),
               'unknown_tracks': sum(t['decision'] == UNDET for t in result['tracks'].values()),
               'untracked_retention': result['untracked_head_retention'],
               'new_classifier_crops': classifier.misses - before_misses,
               'new_classifier_seconds': classifier.forward_seconds - before_forward,
               'cached_replay_wall_seconds': time.perf_counter() - started,
               'scope': 'Frozen detector and cached baseline crops, real classifier for new crops. Replay timing includes cache lookups and is not live inferenceFPS.'}
        result.update(hi=.6, lo=.4, media={'source': info, 'stride': stride, 'output_mode': 'full'})
        save(output / f'{name}_predictions.json', result)
        rows.append(row)
        save(output / 'report.json', {'rows': rows})
        print(json.dumps(row, ensure_ascii=True), flush=True)
    audit_hats(base, output / 'hat_audit', baseline_folder)
    return rows


def audit_hats(classifier, output, baseline_folder):
    manifest = json.loads((ROOT / 'data/benchmark/crowded_20261106.json').read_text(encoding='utf8'))
    old = json.loads((Path(baseline_folder) / 'hard_cases/predictions.json').read_text(encoding='utf8'))
    images = {row['id']: cv2.imread(str(ROOT / row['image'])) for row in manifest['images']}
    items = []
    pixels = []
    for case, result in old.items():
        heads = [head for _, records in result['per_frame'] for _, _, _, boxes in records for head, _ in boxes]
        for index, head in enumerate(heads):
            for margin in [0., .05, .15, .3, .5]:
                box = square(head, margin=margin)
                image = crop(images[case], box, 32)
                if image is None:
                    continue
                items.append({'case': case, 'head_index': index, 'head_box': head, 'crop_box': box,
                              'margin': margin, 'crop_shape': list(image.shape)})
                pixels.append(image)
    probabilities = classifier(pixels)
    for row, probability in zip(items, probabilities):
        row.update(p_no_helmet=probability, decision=decide([probability], 1)[0])
    save(output / 'crop_margin_predictions.json', {'scope': 'Exploratory crop audit on development images, not a fitted threshold or held-out accuracy estimate', 'rows': items})
    print('Fur hat margins:', [(r['head_index'], r['margin'], round(r['p_no_helmet'], 3)) for r in items if r['case'] == 'fur_hat'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--labels', type=Path)
    args = parser.parse_args()
    benchmark(args.source, args.baseline, args.out, args.device, args.labels)
