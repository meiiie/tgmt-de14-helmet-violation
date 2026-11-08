"""Head geometry must survive disabling CLIP; toggles must not reload common models."""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from de14.runtime import RuntimeConfig, build_runtime
from de14.pipeline import Pipeline


class HeadModesTest(unittest.TestCase):
    def config(self, **kwargs):
        try: return RuntimeConfig(device='cpu', detect_heads=True, **kwargs)
        except TypeError as error: self.fail(f'Head localization must be independent of CLIP: {error}')

    def models(self):
        body = lambda _: {'motorcycle': [[20, 70, 180, 190]], 'motorcycle_score': [.9],
                          'person': [[30, 10, 120, 180]], 'person_score': [.9]}
        heads = lambda _: [(.9, [40, 10, 75, 45]), (.8, [100, 50, 135, 85])]
        base = lambda cs: [.1]*len(cs)
        return body, heads, base

    def test_disabling_clip_keeps_independent_child_head_boxes(self):
        body, heads, base = self.models()
        with patch('de14.detectors.YoloDetector', return_value=body), \
             patch('de14.headdet.DeimHeadDetector', return_value=heads), \
             patch('de14.classifiers.DinoProbe.load', return_value=base), \
             patch('de14.classifiers.ClipHelmetVerifier', side_effect=AssertionError('CLIP should not load')):
            bundle = build_runtime(self.config(accurate=False))
            r = bundle.pipeline(fps=1, n_min=1).run([(0, np.zeros((200, 200, 3), np.uint8))], tracking=False)
        self.assertEqual([h for h, _ in r['per_frame'][0][1][0][3]], [[40, 10, 75, 45], [100, 50, 135, 85]])
        self.assertEqual(next(iter(r['tracks'].values()))['n_riders'], 2)

    def test_switching_clip_reuses_detector_and_base_classifier(self):
        body, heads, base = self.models()
        with patch('de14.detectors.YoloDetector', return_value=body), \
             patch('de14.headdet.DeimHeadDetector', return_value=heads), \
             patch('de14.classifiers.DinoProbe.load', return_value=base), \
             patch('de14.classifiers.ClipHelmetVerifier', return_value=lambda cs: [-3.]*len(cs)):
            first = build_runtime(self.config(accurate=False))
        # Any repeated common-model constructor here means toggle reuse failed.
        with patch('de14.detectors.YoloDetector', side_effect=AssertionError('YOLO reloaded')), \
             patch('de14.headdet.DeimHeadDetector', side_effect=AssertionError('head model reloaded')), \
             patch('de14.classifiers.DinoProbe.load', side_effect=AssertionError('DINO reloaded')), \
             patch('de14.classifiers.ClipHelmetVerifier', return_value=lambda cs: [-3.]*len(cs)):
            try: second = build_runtime(replace(first.config, accurate=True), reuse=first)
            except TypeError as error: self.fail(f'Runtime sharing is missing: {error}')
        self.assertIs(first.detector, second.detector)
        self.assertIs(first.classifier.base, second.classifier.base)
        self.assertEqual(second.classifier([np.zeros((50, 50, 3), np.uint8)]), [.1])

    def test_changed_image_size_builds_a_new_detector(self):
        body, heads, base = self.models()
        with patch('de14.detectors.YoloDetector', side_effect=[body, lambda _: {}]), \
             patch('de14.headdet.DeimHeadDetector', return_value=heads), \
             patch('de14.classifiers.DinoProbe.load', return_value=base):
            first = build_runtime(self.config(accurate=False))
            try: second = build_runtime(replace(first.config, imgsz=960), reuse=first)
            except TypeError as error: self.fail(f'Runtime sharing is missing: {error}')
        self.assertIsNot(first.detector, second.detector)

    def test_rendering_one_image_does_not_encode_a_video(self):
        d = {'motorcycle': [[20, 70, 180, 190]], 'motorcycle_score': [.9],
             'person': [], 'head': [[40, 10, 75, 45]], 'head_detection': True}
        image = np.zeros((200, 200, 3), np.uint8)
        p = Pipeline(lambda _: d, lambda cs: [.1]*len(cs), n_min=1)
        with tempfile.TemporaryDirectory() as folder, \
             patch('cv2.VideoWriter', side_effect=AssertionError('Image must not create MP4')):
            try: result = p.run([(0, image)], out_dir=folder, write_video=False, render=True, tracking=False)
            except TypeError as error: self.fail(f'Independent image rendering is missing: {error}')
            self.assertEqual(p.last_vis.shape, image.shape)
            self.assertTrue((Path(folder)/'video_tracks.csv').is_file())
            self.assertEqual(list(Path(folder).glob('*.mp4')), [])
        self.assertEqual(next(iter(result['tracks'].values()))['n_riders'], 1)


if __name__ == '__main__': unittest.main()
