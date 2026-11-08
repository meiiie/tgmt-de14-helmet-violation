"""Video sampling must not shorten the rendered timeline or discard audio."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from de14.pipeline import Pipeline


class VideoTimelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def source(self, count=1050, fps=30, audio=False):
        path = self.root / 'source.mp4'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (96, 64))
        for i in range(count):
            image = np.full((64, 96, 3), 30 if i < count-1 else 220, np.uint8)
            writer.write(image)
        writer.release()
        if audio:
            target = self.root / 'with_audio.mp4'
            subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(path), '-f', 'lavfi',
                            '-i', f'sine=frequency=440:duration={count/fps}', '-map', '0:v',
                            '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', str(target)], check=True)
            return target
        return path

    def runner(self, source, **kwargs):
        from de14.video_io import run_video
        empty = lambda _: {'motorcycle': [], 'person': []}
        return run_video(source, Pipeline(empty, lambda _: []), self.root / 'out', **kwargs)

    def test_eighty_inferences_still_write_all_1050_frames(self):
        result = self.runner(self.source(), stride=3, max_seconds=8)
        media = result['media']
        self.assertEqual(result['frames'], 80)
        self.assertEqual(media['written_frames'], 1050)
        self.assertEqual(media['output']['frame_count'], 1050)
        self.assertEqual(media['output']['fps_rational'], '30/1')
        self.assertEqual((media['output']['width'], media['output']['height']), (96, 64))
        self.assertAlmostEqual(media['output']['video_duration'], 35, places=3)
        cap = cv2.VideoCapture(result['video_path'])
        first_ok, first = cap.read()
        last = first
        while True:
            ok, image = cap.read()
            if not ok: break
            last = image
        cap.release()
        self.assertTrue(first_ok)
        self.assertLess(first.mean(), 50)
        self.assertGreater(last.mean(), 200)
        self.assertEqual(result['timeline'][79]['source_index'], 237)
        self.assertAlmostEqual(result['timeline'][79]['timestamp_s'], 7.9)

    def test_default_analyzes_the_tail_and_preserves_source_audio(self):
        result = self.runner(self.source(32, audio=True), stride=10)
        self.assertEqual(result['frames'], 4)
        self.assertEqual(result['media']['output']['frame_count'], 32)
        self.assertEqual(result['media']['output']['audio'][0]['codec_name'], 'aac')
        self.assertEqual(result['timeline'][3]['source_index'], 30)

    def test_preview_is_explicit_and_keeps_a_distinct_fps_and_name(self):
        result = self.runner(self.source(32), stride=10, output_mode='sampled_preview')
        self.assertEqual(result['media']['output']['frame_count'], 4)
        self.assertEqual(result['media']['output']['fps_rational'], '3/1')
        self.assertIn('sampled_preview', result['video_path'])

    def test_sampling_uses_source_indices_at_stride_two_five_ten(self):
        from de14.video_io import probe_video, sampled_frames
        source = self.source(1052)
        info = probe_video(source)
        for stride, count, last in [(2, 526, 1050), (5, 211, 1050), (10, 106, 1050)]:
            timeline = {}
            rows = list(sampled_frames(source, info, stride=stride, timeline=timeline))
            self.assertEqual(len(rows), count)
            self.assertEqual(timeline[count-1]['source_index'], last)
            self.assertAlmostEqual(timeline[1]['timestamp_s'], stride/30, places=6)

    def test_variable_frame_rate_preserves_presentation_timestamps_and_audio(self):
        from de14.video_io import probe_video
        source = self.source(12, audio=True)
        vfr = self.root / 'vfr.mp4'
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(source), '-vf',
                        "setpts='if(lt(N,6),N,2*N-6)/(30*TB)'", '-fps_mode', 'vfr',
                        '-c:a', 'copy', str(vfr)], check=True)
        original = probe_video(vfr)
        self.assertTrue(original['vfr'])
        result = self.runner(vfr, stride=3)
        media = result['media']
        self.assertEqual(media['output']['timestamps'], original['timestamps'])
        self.assertEqual(media['output']['video_duration'], original['video_duration'])
        self.assertEqual(media['output']['frame_count'], 12)
        self.assertEqual(media['audio_policy'], 'source_audio_copy')
        self.assertEqual(media['output']['audio'][0]['codec_name'], 'aac')
        self.assertEqual(result['timeline'][3]['timestamp_s'], original['timestamps'][9])

    def test_nonzero_source_start_does_not_shift_audio_out_of_the_video(self):
        source = self.source(60, audio=True)
        shifted = self.root / 'shifted.mp4'
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(source), '-c', 'copy',
                        '-output_ts_offset', '5', str(shifted)], check=True)
        result = self.runner(shifted, stride=10)
        decoded = subprocess.run(['ffmpeg', '-v', 'error', '-i', result['video_path'], '-map', '0:a:0',
                                  '-f', 's16le', '-acodec', 'pcm_s16le', 'pipe:1'], check=True, capture_output=True).stdout
        self.assertGreater(len(decoded), 150000)  # 2s mono 44.1kHz/16-bit ~176kB

    def test_rotation_metadata_cannot_silently_scramble_source_pixels(self):
        from de14.video_io import probe_video
        source = self.source(12)
        rotated = self.root / 'rotated.mp4'
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-display_rotation:v:0', '90',
                        '-i', str(source), '-c', 'copy', str(rotated)], check=True)
        with self.assertRaisesRegex(ValueError, 'rotation'):
            probe_video(rotated)

    def test_export_paths_cannot_overwrite_or_delete_the_source(self):
        from de14.video_io import probe_video, render_video
        source = self.source(12)
        result = {'timeline': {}, 'per_frame': [], 'frame_decisions': {}, 'tracks': {}, 'media': {'stride': 1}}
        for source_name, output_name in [('source.mp4', 'source.mp4'), ('target.partial.mp4', 'target.mp4')]:
            with self.subTest(source=source_name):
                target_source = self.root/source_name
                if source != target_source: source.rename(target_source)
                original = target_source.read_bytes()
                with self.assertRaisesRegex(ValueError, 'source'):
                    render_video(target_source, probe_video(target_source), result, self.root/output_name)
                self.assertEqual(target_source.read_bytes(), original)
                source = target_source


if __name__ == '__main__': unittest.main()
