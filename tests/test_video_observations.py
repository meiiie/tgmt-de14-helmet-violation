"""Sampling/rendering must not create votes, hide F2 removals or lose source time."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from de14.core import UNDET, VIOL
from de14.pipeline import Pipeline


class VideoObservationsTest(unittest.TestCase):
    def run_case(self, count, probability=.9, heads=True, filtering=None, metadata=None):
        d = {'motorcycle': [[20, 70, 180, 190]], 'motorcycle_score': [.9], 'person': [],
             'head': [[30, 10, 60, 40], [70, 20, 100, 50], [110, 30, 140, 60], [140, 40, 165, 65]] if heads else [],
             'head_detection': True}
        return Pipeline(lambda _: d, lambda cs: [probability]*len(cs), fps=3, n_min=10,
                        track_filter=filtering).run(
            [(i, np.zeros((200, 200, 3), np.uint8)) for i in range(count)],
            write_video=False, evidence=False, frame_metadata=metadata)

    def test_four_heads_are_one_known_vehicle_observation_per_sample(self):
        nine = self.run_case(9)
        self.assertEqual(nine['tracks'][1]['n_obs'], 9)
        self.assertEqual(nine['tracks'][1]['decision'], UNDET)
        ten = self.run_case(10)
        self.assertEqual(ten['tracks'][1]['n_obs'], 10)
        self.assertEqual(ten['tracks'][1]['decision'], VIOL)
        self.assertEqual(ten['counters']['person_detections'], 0)
        self.assertEqual(ten['counters']['associated_heads'], 40)

    def test_unknown_classification_does_not_count_as_a_known_vote(self):
        result = self.run_case(10, probability=None)
        audit = result['track_audit'][1]
        self.assertEqual(audit['known_vehicle_observations'], 0)
        self.assertEqual(audit['unknown_observations'], 10)
        self.assertEqual(audit['head_positive_frames'], 10)

    def test_f2_keeps_head_evidence_when_person_detector_misses(self):
        result = self.run_case(10, filtering={'tau_disp': .75, 'tau_person_static': .7})
        self.assertEqual(result['f2']['total_before'], 1)
        self.assertEqual(result['f2']['kept'], 1)
        self.assertEqual(result['f2']['removed'], 0)

    def test_f2_records_removed_empty_stationary_vehicle(self):
        result = self.run_case(10, heads=False, filtering={'tau_disp': .75, 'tau_person_static': .7})
        self.assertEqual(result['tracks'], {})
        self.assertEqual(result['f2'], {'total_before': 1, 'removed': 1, 'kept': 0})
        self.assertEqual(result['track_audit'][1]['reason'], 'f2_low_displacement_low_rider_evidence')
        self.assertEqual(len(result['per_frame']), 10)

    def test_causal_votes_and_csv_use_source_time_without_future_labels(self):
        metadata = {i: {'source_index': i*10, 'timestamp_s': i/3, 'inference_index': i} for i in range(10)}
        result = self.run_case(10, metadata=metadata)
        self.assertEqual(result['frame_decisions'][8][1], UNDET)
        self.assertEqual(result['frame_decisions'][9][1], VIOL)
        self.assertEqual(result['observations'][1][-1]['source_index'], 90)
        self.assertEqual(result['observations'][1][-1]['t'], 3)
        self.assertEqual(result['track_audit'][1]['track_detection_frames'], 10)
        self.assertEqual(result['tracking_audit'][-1]['source_index'], 90)
        self.assertEqual(result['tracking_audit'][-1]['lost_buffer_updates'], 30)
        self.assertEqual(result['tracking_audit'][-1]['lost_buffer_seconds'], 10)


if __name__ == '__main__': unittest.main()
