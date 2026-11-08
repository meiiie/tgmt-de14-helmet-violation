"""Shared CPU/CUDA video runner: sampled inference, full source timeline rendering.

Unsampled frames are passed through without boxes. No interpolation, synthetic
observations, tracker updates or classification results are created by rendering.
FFmpeg/ffprobe are required. Full VFR exports preserve presentation timestamps.
"""
from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import subprocess
import tempfile
import time

import cv2
import numpy as np

from .pipeline import draw_annotations


def json_default(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def probe_video(path):
    """Inspect decoded timestamps rather than trusting container frame estimates."""
    command = ['ffprobe', '-v', 'error', '-threads', '4', '-show_streams', '-show_format',
               '-show_frames', '-show_entries',
               'frame=media_type,best_effort_timestamp,duration,pkt_duration:stream:format',
               '-of', 'json', str(path)]
    data = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
    streams = [s for s in data['streams'] if s['codec_type'] == 'video']
    if len(streams) != 1: raise ValueError('Exactly one video stream is required')
    stream = streams[0]
    rotations = [s.get('rotation', 0) for s in stream.get('side_data_list', [])]
    rotations.append(stream.get('tags', {}).get('rotate', 0))
    if any(float(rotation) % 360 for rotation in rotations):
        raise ValueError('Video display rotation is not supported; normalize orientation explicitly before analysis.')
    frames = [f for f in data.get('frames', []) if f.get('media_type') == 'video']
    if not frames or any('best_effort_timestamp' not in f for f in frames):
        raise ValueError('Missing decoded video timestamps')
    time_base = Fraction(stream['time_base'])
    origin = int(frames[0]['best_effort_timestamp'])
    pts = [int(f['best_effort_timestamp']) - origin for f in frames]
    intervals = [b-a for a, b in zip(pts, pts[1:])]
    if any(delta <= 0 for delta in intervals):
        raise ValueError('Source video timestamps are not strictly increasing')
    nominal = stream.get('r_frame_rate')
    fps = Fraction(nominal) if nominal and nominal != '0/0' else Fraction(0)
    last_duration = frames[-1].get('duration') or frames[-1].get('pkt_duration') or 0
    last_ticks = int(last_duration) if last_duration != 'N/A' else 0
    if last_ticks <= 0:
        last_ticks = intervals[-1] if intervals else round(1 / (fps * time_base)) if fps else 0
    if last_ticks <= 0:
        raise ValueError('Cannot determine source video duration')
    duration = (pts[-1] + last_ticks) * time_base
    if fps:
        interval = float(1/fps)
        tolerance = max(.0001, interval*.002)
        vfr = any(abs(float(delta*time_base)-interval) > tolerance for delta in (*intervals, last_ticks))
    else:
        vfr = True
    if vfr:
        # Containers can advertise a rate far above the actual number of frames.
        # Use the observed average for tracker sizing and CFR previews only.
        fps = len(pts) / duration
    else:
        duration = len(pts) / fps
    return {'path': str(Path(path).resolve()), 'fps': float(fps), 'fps_rational': str(fps.numerator)+'/'+str(fps.denominator),
            'frame_count': len(pts), 'width': int(stream['width']), 'height': int(stream['height']),
            'video_duration': float(duration), 'container_duration': float(data['format']['duration']),
            'video_start_time': float(origin * time_base), 'timestamps': [float(t*time_base) for t in pts],
            'vfr': vfr, 'time_base': str(time_base), 'pts_ticks': pts, 'last_frame_ticks': last_ticks,
            'audio': [{k: s[k] for k in ('codec_name', 'sample_rate', 'channels', 'start_time', 'duration') if k in s}
                      for s in data['streams'] if s['codec_type'] == 'audio']}


def sampled_frames(path, info, stride=3, max_seconds=0, max_frames=0, timeline=None, stats=None):
    if stride < 1 or max_seconds < 0 or max_frames < 0: raise ValueError('Invalid sampling limits')
    timeline = timeline if timeline is not None else {}
    stats = stats if stats is not None else {}
    cap = cv2.VideoCapture(str(path)); i = k = 0
    stats.update(decoded_frames=0, decode_seconds=0., stop_reason='source_end')
    try:
        if not cap.isOpened(): raise ValueError(f'Cannot open video: {path}')
        while i < info['frame_count']:
            timestamp = info['timestamps'][i]
            if max_seconds and timestamp >= max_seconds:
                stats['stop_reason'] = 'analysis_seconds_limit'; break
            if max_frames and k >= max_frames:
                stats['stop_reason'] = 'analysis_sample_limit'; break
            start = time.perf_counter(); ok, image = cap.read()
            stats['decode_seconds'] += time.perf_counter()-start
            if not ok: raise ValueError(f'Video decode ended early at source frame {i}')
            stats['decoded_frames'] += 1
            if i % stride == 0:
                timeline[k] = {'source_index': i, 'timestamp_s': timestamp, 'inference_index': k}
                yield k, image
                k += 1
            i += 1
    finally:
        cap.release()


def _encode_cfr(source, info, frames, target, error_path, fps, audio, preset, pixel_format):
    command = ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pixel_format', 'bgr24',
               '-video_size', f"{info['width']}x{info['height']}", '-framerate', str(fps), '-i', 'pipe:0']
    if audio:
        # Normalize source video PTS to zero; shift audio by the same source origin.
        command += ['-copyts', '-itsoffset', str(-info['video_start_time']), '-i', str(source)]
    command += ['-map', '0:v:0']
    if audio: command += ['-map', '1:a', '-c:a', 'copy']
    command += ['-c:v', 'libx264', '-preset', preset, '-crf', '20', '-pix_fmt', pixel_format,
                '-fps_mode', 'passthrough', '-movflags', '+faststart', str(target)]
    # A file for stderr avoids a pipe-buffer deadlock on long encoder errors.
    with error_path.open('wb') as errors:
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
        try:
            for _, image in frames:
                encoder.stdin.write(np.ascontiguousarray(image).tobytes())
        finally:
            encoder.stdin.close()
            encoder.wait()
    if encoder.returncode:
        raise RuntimeError(f'FFmpeg export failed: {error_path.read_text(errors="replace")}')


def _encode_vfr(source, info, frames, target, error_path, fps, preset, pixel_format):
    import av

    time_base = Fraction(info['time_base'])
    pts = info['pts_ticks']
    durations = dict(zip(pts, [b-a for a, b in zip(pts, pts[1:])] + [info['last_frame_ticks']]))
    mux_options = {'video_track_timescale': str(time_base.denominator)}
    with tempfile.TemporaryDirectory(prefix='de14_vfr_', dir=target.parent) as folder:
        video = Path(folder) / 'video.mp4' if info['audio'] else target
        with av.open(str(video), mode='w', options={**mux_options, 'movflags': '+faststart'}) as container:
            stream = container.add_stream('libx264', rate=fps.limit_denominator(1000))
            stream.width, stream.height = info['width'], info['height']
            stream.pix_fmt = pixel_format
            stream.time_base = stream.codec_context.time_base = time_base
            # Keep decode order equal to presentation order so MP4 sample durations
            # follow the original per-frame intervals, including the final frame.
            stream.codec_context.max_b_frames = 0
            stream.options = {'preset': preset, 'crf': '20'}

            def mux(packets):
                for packet in packets:
                    packet.duration = durations[packet.pts]
                    container.mux(packet)

            for index, image in frames:
                frame = av.VideoFrame.from_ndarray(image, format='bgr24')
                frame.pts, frame.time_base = pts[index], time_base
                mux(stream.encode(frame))
            mux(stream.encode())

        audio_codecs = []
        if info['audio']:
            command = ['ffmpeg', '-y', '-v', 'error', '-copyts', '-i', str(video),
                       '-itsoffset', str(-info['video_start_time']), '-i', str(source),
                       '-map', '0:v:0', '-map', '1:a', '-c:v', 'copy']
            for index, audio_stream in enumerate(info['audio']):
                codec = 'copy' if audio_stream['codec_name'] in {'aac', 'mp3'} else 'aac'
                audio_codecs.append(codec)
                command += [f'-c:a:{index}', codec]
            command += ['-video_track_timescale', mux_options['video_track_timescale'],
                        '-movflags', '+faststart', str(target)]
            with error_path.open('wb') as errors:
                encoded = subprocess.run(command, stderr=errors)
            if encoded.returncode:
                raise RuntimeError(f'FFmpeg audio mux failed: {error_path.read_text(errors="replace")}')
        return 'aac_for_incompatible_audio' if 'aac' in audio_codecs else 'source_audio_copy' if audio_codecs else 'no_audio'


def render_video(source, info, result, output, output_mode='full', preset='veryfast'):
    """Second streaming decode pass; apply final F2 keep/drop and causal votes."""
    if output_mode not in {'full', 'sampled_preview'}: raise ValueError('Unknown output mode')
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem+'.partial.mp4')
    if Path(source).resolve() in {output.resolve(), temporary.resolve()}:
        raise ValueError('Export output and temporary paths must differ from the source video')
    if preset not in {'fast', 'veryfast'}:
        raise ValueError('Unsupported encoder preset')
    records = {result['timeline'][fi]['source_index']: (fi, rec) for fi, rec in result['per_frame']}
    untracked_records = result.get('untracked_detections', {})
    fps = Fraction(info['fps_rational'])
    if output_mode == 'sampled_preview': fps /= result['media']['stride']
    audio = bool(info['audio']) and output_mode == 'full'
    vfr = info.get('vfr', False) and output_mode == 'full'
    pixel_format = 'yuv420p' if info['width'] % 2 == 0 and info['height'] % 2 == 0 else 'yuv444p'
    cap = cv2.VideoCapture(str(source)); written = decoded = 0
    start = time.perf_counter()
    error_path = output.with_suffix('.ffmpeg.log')

    def frames():
        nonlocal written, decoded
        while True:
            ok, image = cap.read()
            if not ok: break
            source_index = decoded; decoded += 1
            if output_mode == 'sampled_preview' and source_index not in records: continue
            if source_index in records:
                fi, rec = records[source_index]
                image = draw_annotations(image, rec, result['frame_decisions'][fi],
                                         result['timeline'][fi]['timestamp_s'], result['tracks'],
                                         result.get('hi', .6), result.get('lo', .4),
                                         untracked=untracked_records.get(fi, untracked_records.get(str(fi), [])))
            written += 1
            yield source_index, image

    try:
        if vfr:
            audio_policy = _encode_vfr(source, info, frames(), temporary, error_path, fps, preset, pixel_format)
        else:
            _encode_cfr(source, info, frames(), temporary, error_path, fps, audio, preset, pixel_format)
            audio_policy = 'source_audio_copy' if audio else 'no_audio' if not info['audio'] else 'preview_audio_omitted'
        if decoded != info['frame_count']: raise ValueError('Render decode frame count differs from source')
        exported = probe_video(temporary)
        expected = info['frame_count'] if output_mode == 'full' else len(records)
        if (written != expected or exported['frame_count'] != expected
                or (exported['width'], exported['height']) != (info['width'], info['height'])):
            raise ValueError('Export validation failed: frame count/resolution mismatch')
        if vfr:
            source_tb, output_tb = Fraction(info['time_base']), Fraction(exported['time_base'])
            if (any(a*source_tb != b*output_tb for a, b in zip(info['pts_ticks'], exported['pts_ticks']))
                    or info['last_frame_ticks']*source_tb != exported['last_frame_ticks']*output_tb):
                raise ValueError('VFR export did not preserve source timestamps or final frame duration')
        elif Fraction(exported['fps_rational']) != fps:
            raise ValueError('Export FPS mismatch')
        if not vfr and abs(exported['video_duration']-expected/float(fps)) > 1/float(fps):
            raise ValueError('Export duration mismatch')
        if audio and len(exported['audio']) != len(info['audio']): raise ValueError('Source audio was not retained')
        temporary.replace(output)
        exported['path'] = str(output.resolve())
        result['media'].update(written_frames=written, render_decoded_frames=decoded, output=exported,
                               audio_policy=audio_policy,
                               render_backend='pyav_pts' if vfr else 'raw_bgr_pipe', encoder_preset=preset,
                               render_encode_seconds=time.perf_counter()-start)
        return str(output.resolve())
    finally:
        cap.release()
        temporary.unlink(missing_ok=True)


def run_video(source, pipeline, out_dir, video_name='video', stride=3, max_seconds=0, max_frames=0,
              output_mode='full', trace=True):
    """Same runner for Gradio, CLI and benchmarks; max limits affect inference only."""
    start = time.perf_counter()
    if stride < 1 or max_seconds < 0 or max_frames < 0: raise ValueError('Invalid sampling limits')
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    info = probe_video(source); probe_seconds = time.perf_counter()-start
    pipeline.fps = info['fps']/stride
    timeline = {}; decode = {}
    def emit(event):
        meta = timeline.get(event.get('frame'), {})
        audit.write(json.dumps({**event, **meta}, default=json_default, ensure_ascii=False, allow_nan=False)+'\n')
    with (out_dir/f'{video_name}_stages.jsonl').open('w', encoding='utf-8') as audit:
        result = pipeline.run(sampled_frames(source, info, stride, max_seconds, max_frames, timeline, decode),
                              out_dir=str(out_dir), video_name=video_name, write_video=False,
                              frame_metadata=timeline, trace=emit if trace else None)
    result['media'] = {'source': info, 'stride': stride, 'sample_interval_s': stride/info['fps'],
                       'output_mode': output_mode, 'max_seconds': max_seconds, 'max_frames': max_frames,
                       'sampled_frames': result['frames'], 'probe_seconds': probe_seconds, **decode,
                       'analyzed_last_timestamp_s': max((m['timestamp_s'] for m in timeline.values()), default=None),
                       'unsampled_policy': 'original_unannotated', 'interpolation': False}
    result.update(hi=pipeline.hi, lo=pipeline.lo)
    suffix = 'annotated' if output_mode == 'full' else 'sampled_preview'
    result['video_path'] = render_video(source, info, result, out_dir/f'{video_name}_{suffix}.mp4', output_mode)
    result['total_seconds'] = time.perf_counter()-start
    with (out_dir/f'{video_name}_audit.json').open('w', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=2, default=json_default, allow_nan=False)
    return result
