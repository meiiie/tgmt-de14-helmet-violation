"""App/CLI must return full media even when inference is explicitly capped."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'src'))
import app
import run_video as cli
from de14.runtime import RuntimeBundle, RuntimeConfig
from de14.video_io import probe_video


class VideoEntrypointsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.source = self.root/'source.mp4'
        writer = cv2.VideoWriter(str(self.source), cv2.VideoWriter_fourcc(*'mp4v'), 30, (96,64))
        for _ in range(60): writer.write(np.full((64,96,3), 100, np.uint8))
        writer.release()
        empty = lambda _: {'motorcycle': [], 'person': []}
        # Only the expensive model boundary is replaced; inference loop/media/exports are real.
        self.runtime = RuntimeBundle(empty, lambda _: [], RuntimeConfig(device='cpu', accurate=False, detect_heads=True), 'cpu')

    def assert_full_source(self, path):
        info = probe_video(path)
        self.assertEqual(info['frame_count'], 60)
        self.assertEqual(info['fps_rational'], '30/1')
        self.assertEqual(info['video_duration'], 2)

    def test_app_analysis_limit_does_not_crop_export(self):
        with patch.object(app, 'get_runtime', return_value=self.runtime), \
             patch.object(app.tempfile, 'mkdtemp', return_value=str(self.root/'app_output')):
            result = app.analyze(None, str(self.source), app.BACKENDS[1], 3, 1, 5, 1280, True, False, 'openvino_cpu')
        self.assert_full_source(result[1])

    def test_cli_sample_limit_does_not_crop_export(self):
        output = self.root/'cli_output'
        argv = ['run_video.py', str(self.source), '--profile', 'openvino_cpu', '--no-accuracy',
                '--stride', '10', '--max_frames', '2', '--out', str(output)]
        with patch.object(cli, 'build_runtime', return_value=self.runtime), patch.object(sys, 'argv', argv):
            cli.main()
        self.assert_full_source(output/'source_annotated.mp4')


if __name__ == '__main__': unittest.main()
