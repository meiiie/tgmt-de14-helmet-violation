"""Measured heads remain usable while vehicle identity is unconfirmed."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from de14.pipeline import Pipeline
from de14.core import VIOL, UNDET

EMPTY = {'motorcycle': [], 'person': [], 'head': [], 'head_detection': True}
SCENE = {'motorcycle': [[20, 70, 180, 190]], 'motorcycle_score': [.9], 'person': [],
         'head': [[40, 10, 70, 40]], 'head_detection': True}


class UntrackedHeadTest(unittest.TestCase):
    def run_scene(self, scenes, values):
        sequence = iter(scenes)
        calls = []
        def classify(crops):
            calls.append([int(c[0, 0, 0]) for c in crops])
            return [.9 if c[0, 0, 0] < 100 else .1 for c in crops]
        pipeline = Pipeline(lambda _: next(sequence), classify, batch_heads=True,
                            retain_untracked_heads=True, n_min=2)
        result = pipeline.run([(i, np.full((200, 200, 3), value, np.uint8)) for i, value in enumerate(values)],
                              write_video=False, evidence=False)
        return result, calls

    def test_low_score_vehicle_head_is_cropped_without_inventing_track_votes(self):
        scene = {**SCENE, 'motorcycle_score': [.35]}
        result, calls = self.run_scene([scene, scene], [11, 22])
        self.assertEqual(calls, [[11], [22]])
        self.assertEqual(result['tracks'], {})
        self.assertEqual(sum(len(values) for values in result['observations'].values()), 0)
        self.assertEqual([len(result['untracked_detections'][i]) for i in range(2)], [1, 1])
        self.assertEqual(result['untracked_detections'][0][0]['heads'][0][1], .9)
        self.assertIsNone(result['untracked_detections'][0][0]['candidate_track_id'])

    def test_confirmation_reuses_past_prediction_and_removes_duplicate_overlay(self):
        result, calls = self.run_scene([EMPTY, SCENE, SCENE], [0, 11, 200])
        self.assertEqual(calls, [[11], [200]])
        tid = next(iter(result['tracks']))
        self.assertEqual([o['p_no_helmet'] for o in result['observations'][tid]], [.9, .1])
        self.assertEqual(result['untracked_detections'][1], [])
        self.assertEqual(result['frame_decisions'][1][tid], UNDET)
        self.assertEqual(result['frame_decisions'][2][tid], UNDET)

    def test_removed_candidate_does_not_transfer_a_prediction_to_new_identity(self):
        result, calls = self.run_scene([EMPTY, SCENE, EMPTY, SCENE, SCENE], [0, 7, 0, 22, 200])
        self.assertEqual(calls, [[7], [22], [200]])
        self.assertEqual(result['untracked_detections'][1][0]['heads'][0][1], .9)
        obs = next(iter(result['observations'].values()))
        self.assertEqual([o['frame'] for o in obs], [3, 4])
        self.assertEqual([o['p_no_helmet'] for o in obs], [.9, .1])

    def test_runtime_factory_enables_measured_head_retention(self):
        from de14.runtime import RuntimeBundle, RuntimeConfig
        runtime = RuntimeBundle(None, None, RuntimeConfig(), 'cpu')
        self.assertTrue(runtime.pipeline().retain_untracked_heads)
        self.assertFalse(runtime.pipeline(retain_untracked_heads=False).retain_untracked_heads)

    def test_live_render_shows_measured_head_while_track_report_stays_unconfirmed(self):
        scene = {**SCENE, 'motorcycle_score': [.35]}
        pipeline = Pipeline(lambda _: scene, lambda crops: [.9] * len(crops), retain_untracked_heads=True)
        result = pipeline.run([(0, np.zeros((200, 200, 3), np.uint8))],
                              write_video=False, evidence=False, render=True)
        self.assertEqual(result['tracks'], {})
        self.assertEqual(sum(len(values) for values in result['observations'].values()), 0)
        self.assertGreater(int(pipeline.last_vis[10, 40, 2]), 200)

    def test_unknown_tentative_prediction_is_reused_without_reclassification(self):
        sequence = iter([EMPTY, SCENE, SCENE])
        calls = []
        def classify(crops):
            calls.append(len(crops))
            return [None] * len(crops)
        pipeline = Pipeline(lambda _: next(sequence), classify, batch_heads=True, retain_untracked_heads=True)
        result = pipeline.run([(i, np.zeros((200, 200, 3), np.uint8)) for i in range(3)],
                              write_video=False, evidence=False)
        self.assertEqual(calls, [1, 1])
        self.assertEqual(result['untracked_head_retention']['reused_predictions'], 1)
        self.assertEqual(result['untracked_detections'][1], [])
        self.assertEqual(sum(len(values) for values in result['observations'].values()), 0)


if __name__ == '__main__':
    unittest.main()
