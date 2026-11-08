"""Vote replay must reuse measurements without silently counting unknown heads."""
import copy
from pathlib import Path
import sys
import unittest
import tempfile
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from de14.core import UNDET, VIOL


class VideoBenchmarkTest(unittest.TestCase):
    def test_encoder_weights_and_preprocessing_are_part_of_resume_identity(self):
        from benchmark_video_sampling import encoder_identity
        import torch
        model = torch.nn.Linear(2, 1)
        processor = {'size': 224, 'normalize': True}
        encoder = SimpleNamespace(model=model, proc=SimpleNamespace(to_dict=lambda: dict(processor)), model_id='test')
        original = encoder_identity(encoder)
        with torch.no_grad(): model.weight.add_(1)
        self.assertNotEqual(original, encoder_identity(encoder))
        weighted = encoder_identity(encoder)
        processor['size'] = 256
        self.assertNotEqual(weighted, encoder_identity(encoder))

    def test_rejected_resume_preserves_original_manifest(self):
        from benchmark_video_sampling import validate_provenance
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'provenance.json'
            path.write_bytes(b'{"key":"old","evidence":"retain me"}')
            with self.assertRaisesRegex(ValueError, 'provenance'):
                validate_provenance(path, {'key': 'new'}, resume=True)
            self.assertEqual(path.read_bytes(), b'{"key":"old","evidence":"retain me"}')

    def test_replay_changes_only_votes_and_keeps_f2_drops_auditable(self):
        from benchmark_video_sampling import replay_votes, summarize
        result = {'tracks': {1: {'decision': UNDET, 'n_obs': 6}},
                  'observations': {1: [{'frame': i, 'p_no_helmet': .9} for i in range(6)]},
                  'track_audit': {1: {'kept': True, 'decision_before_filter': UNDET, 'known_vehicle_observations': 6},
                                  2: {'kept': False, 'decision_before_filter': UNDET, 'known_vehicle_observations': 0}},
                  'per_frame': [(i, [(1, [0, 0, 100, 100], .9, [([0, 0, 10, 10], .9)])]) for i in range(6)],
                  'frames': 6, 'frame_decisions': {}, 'f2': {'total_before': 2, 'kept': 1, 'removed': 1},
                  'counters': {'person_detections': 9}, 'sec': 2, 't_det': 1, 't_cls': .5, 'tracking_audit': []}
        original = copy.deepcopy(result)
        five = replay_votes(result, 5)
        ten = replay_votes(result, 10)
        self.assertEqual(five['tracks'][1]['decision'], VIOL)
        self.assertEqual(ten['tracks'][1]['decision'], UNDET)
        self.assertEqual(five['frame_decisions'][3][1], UNDET)
        self.assertEqual(five['frame_decisions'][4][1], VIOL)
        self.assertEqual(result, original)
        row = summarize(ten, 10, 10)
        self.assertEqual(row['unknown_rate'], 1)
        self.assertEqual(row['f2'], {'total_before': 2, 'kept': 1, 'removed': 1})
        self.assertEqual(row['precision'], None)
        self.assertEqual(row['id_switches'], None)
        self.assertEqual(row['person_detections'], 9)
        self.assertEqual(row['f2_shadow_off']['vehicle_tracks'], 2)


if __name__ == '__main__': unittest.main()
