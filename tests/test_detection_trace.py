"""Stage evidence must account for objects rejected by confidence, tracking and crop size."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from de14.pipeline import Pipeline


class DetectionTraceTest(unittest.TestCase):
    def test_overlap_sweep_preserves_json_serializable_suppression_evidence(self):
        import torch
        import de14.pipeline as module
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from debug_rider_pipeline import sweeps
        raw = np.zeros((1, 84, 2), np.float32)
        raw[0, :4] = [[30, 40], [40, 40], [40, 40], [40, 40]]
        raw[0, 7] = [.9, .8]  # class 3; IoU .6 survives YOLO .7, rejected by custom .5
        d = {'motorcycle': [[10, 20, 50, 60], [20, 20, 60, 60]], 'motorcycle_score': [.9, .8],
             'person': [], 'person_score': [], 'head': [], 'head_score': [], 'head_detection': True}
        torch.set_num_threads(4)
        with tempfile.TemporaryDirectory() as folder:
            report = sweeps(np.zeros((100, 100, 3), np.uint8),
                            {'detector': d, 'input_hw': [100, 100], 'end2end': False},
                            {'yolo': raw, 'head': np.zeros((1, 0, 6), np.float32)}, Path(folder), module)
            self.assertTrue(report['motorcycle_nms'][0]['rejected'])
            json.dumps(report, allow_nan=False)

    def test_every_filtered_head_has_stage_evidence(self):
        detections = {
            'motorcycle': [[20, 70, 150, 190], [220, 70, 350, 190], [420, 70, 550, 190]],
            'motorcycle_score': [.9, .35, .2], 'person': [],
            'head': [[45, 10, 65, 30], [70, 20, 71, 21], [245, 10, 265, 30]],
            'head_score': [.9, .8, .9], 'head_detection': True}
        events = []
        pipeline = Pipeline(lambda _: detections, lambda cs: [.1] * len(cs), n_min=1)
        try:
            result = pipeline.run([(7, np.zeros((200, 600, 3), np.uint8))], trace=events.append)
        except TypeError as error:
            self.fail(f'Pipeline must expose filter evidence through trace: {error}')
        stages = {e['stage']: e for e in events}
        self.assertEqual(stages['motorcycle_confidence']['rejected'],
                         [{'index': 2, 'score': .2, 'reason': 'below_conf_moto'}])
        self.assertEqual(stages['tracking']['untracked_indices'], [1])
        self.assertEqual([h['status'] for h in stages['crops']['heads']], ['valid', 'too_small'])
        self.assertEqual([h['probability'] for h in stages['classification']['heads']], [.1, None])
        self.assertEqual(next(iter(result['tracks'].values()))['n_riders'], 2)
        json.dumps(events, allow_nan=False)

    def test_trace_does_not_change_predictions(self):
        d = {'motorcycle': [[20, 70, 180, 190]], 'motorcycle_score': [.9], 'person': [],
             'head': [[40, 10, 70, 40]], 'head_detection': True}
        def run(trace=None):
            p = Pipeline(lambda _: d, lambda cs: [.9] * len(cs), n_min=1)
            try:
                return p.run([(0, np.zeros((200, 200, 3), np.uint8))], trace=trace)
            except TypeError as error:
                self.fail(f'Optional trace is unsupported: {error}')
        before, after = run(), run(lambda event: None)
        self.assertEqual(before['tracks'], after['tracks'])
        self.assertEqual(before['per_frame'], after['per_frame'])


if __name__ == '__main__':
    unittest.main()
