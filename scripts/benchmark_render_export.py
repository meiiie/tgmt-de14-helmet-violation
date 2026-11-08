"""Compare renderers using frozen predictions, without model inference."""
import argparse
import copy
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from de14.video_io import render_video, json_default


def audio_hash(path):
    result = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-map', '0:a:0',
                             '-c:a', 'copy', '-f', 'hash', '-hash', 'sha256', '-'],
                            capture_output=True, text=True, check=True)
    return result.stdout.strip()


def benchmark(source, audit_path, output, repeats=2):
    result = json.loads(Path(audit_path).read_text(encoding='utf8'))
    for key in ('timeline', 'frame_decisions', 'tracks'):
        result[key] = {int(k): v for k, v in result[key].items()}
    result['frame_decisions'] = {fi: {int(k): v for k, v in values.items()}
                                 for fi, values in result['frame_decisions'].items()}
    info = result['media']['source']
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    variants = [('legacy_fast', 'fast'), ('optimized_veryfast', 'veryfast')]
    rows = []
    expected_audio = audio_hash(source) if info['audio'] else None
    for repeat in range(repeats):
        order = variants if repeat % 2 == 0 else list(reversed(variants))
        for name, preset in order:
            measured = copy.deepcopy(result)
            target = output / f'{name}_{repeat + 1}.mp4'
            started = time.perf_counter()
            render_video(source, info, measured, target, preset=preset)
            seconds = time.perf_counter() - started
            if expected_audio and audio_hash(target) != expected_audio:
                raise ValueError('Audio packet data differs from source')
            rows.append({'variant': name, 'repeat': repeat + 1, 'wall_seconds': seconds,
                         'bytes': target.stat().st_size, 'path': str(target), 'media': measured['media']})
            (output / 'progress.json').write_text(json.dumps(rows, default=json_default, indent=2), encoding='utf8')
            print(name, repeat + 1, round(seconds, 3), flush=True)
    fidelity = {}
    reference = output / 'legacy_fast_1.mp4'
    for name, _ in variants[1:]:
        target = output / f'{name}_1.mp4'
        check = subprocess.run(['ffmpeg', '-v', 'info', '-i', str(reference), '-i', str(target),
                                '-lavfi', '[0:v][1:v]ssim', '-an', '-f', 'null', '-'],
                               capture_output=True, text=True, check=True)
        values = re.findall(r'All:([0-9.]+)', check.stderr)
        if not values:
            raise RuntimeError('SSIM measurement missing')
        fidelity[name] = float(values[-1])
    report = {'scope': 'Render-only wall time, frozen predictions and source, alternating variant order',
              'source': str(source), 'audit': str(audit_path), 'rows': rows,
              'ssim_against_legacy_fast': fidelity, 'audio_packet_hash': expected_audio}
    (output / 'report.json').write_text(json.dumps(report, default=json_default, indent=2), encoding='utf8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('audit', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    benchmark(args.source, args.audit, args.out, args.repeats)
