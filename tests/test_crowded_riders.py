"""Geometry fixtures from the crowded-image failure; no weights or downloads."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from de14.pipeline import Pipeline
from de14 import core


class CrowdedRiderTest(unittest.TestCase):
    def test_four_heads_survive_fragmented_motorcycle_and_track_admission(self):
        d = {'motorcycle': [[359, 171, 586, 433], [146, 148, 380, 434]],
             'motorcycle_score': [.526, .393],
             'person': [[251, 70, 383, 415], [291, 54, 478, 412],
                        [168, 58, 341, 415], [369, 119, 497, 415],
                        [320, 5, 366, 58], [279, 4, 331, 69]],
             'person_score': [.75, .75, .58, .21, .59, .58],
             'head': [[398, 117, 473, 192], [306, 42, 379, 122],
                      [248, 64, 319, 140], [178, 40, 249, 114],
                      [318, 2, 361, 40], [287, 0, 323, 40]],
             'head_score': [.90, .90, .90, .89, .79, .63], 'head_detection': True}
        result = Pipeline(lambda _: d, lambda cs: [.1] * len(cs), n_min=1).run(
            [(0, np.zeros((434, 653, 3), np.uint8))])
        self.assertEqual([t['n_riders'] for t in result['tracks'].values()], [4])
        self.assertEqual(len(result['per_frame'][0][1][0][3]), 4)

    def test_background_person_is_not_a_passenger_of_a_large_foreground_bike(self):
        d = {'motorcycle': [[200, 200, 500, 500]], 'motorcycle_score': [.9],
             'person': [[260, 30, 360, 470], [400, 60, 450, 180]],
             'person_score': [.9, .9], 'head_detection': True,
             'head': [[280, 30, 330, 80], [405, 60, 445, 100]]}
        r = Pipeline(lambda _: d, lambda cs: [.1] * len(cs), n_min=1).run(
            [(0, np.zeros((550, 550, 3), np.uint8))])
        self.assertEqual(next(iter(r['tracks'].values()))['n_riders'], 1)

    def test_static_image_does_not_require_temporal_track_confirmation(self):
        d = {'motorcycle': [[20, 70, 180, 190]], 'motorcycle_score': [.35],
             'person': [], 'head': [[40, 10, 70, 40]], 'head_detection': True}
        p = Pipeline(lambda _: d, lambda cs: [.9] * len(cs), n_min=1)
        try:
            r = p.run([(0, np.zeros((200, 200, 3), np.uint8))], tracking=False)
        except TypeError as error:
            self.fail(f'Static images need an explicit non-temporal path: {error}')
        self.assertEqual(next(iter(r['tracks'].values()))['n_riders'], 1)

    def test_static_admission_requires_rider_evidence_for_a_weak_box(self):
        d = {'motorcycle': [[20, 70, 150, 190], [220, 70, 350, 190]],
             'motorcycle_score': [.42, .35], 'person': [],
             'head': [[245, 10, 275, 40]], 'head_detection': True}
        r = Pipeline(lambda _: d, lambda cs: [.1] * len(cs), n_min=1).run(
            [(0, np.zeros((200, 400, 3), np.uint8))], tracking=False)
        self.assertEqual([t['n_riders'] for t in r['tracks'].values()], [1])

    def test_distinct_bikes_are_not_merged_by_box_overlap_alone(self):
        merge = getattr(core, 'merge_motorcycle_fragments', None)
        self.assertTrue(callable(merge), 'Missing fragment reconciliation')
        boxes = [[20, 100, 200, 300], [170, 100, 350, 300]]
        merged, scores, groups = merge(boxes, [.8, .39], [], [])
        self.assertEqual(merged, boxes)
        self.assertEqual(groups, [[0], [1]])
        self.assertEqual(scores, [.8, .39])

    def test_shared_body_does_not_merge_bikes_at_different_depths(self):
        merge = getattr(core, 'merge_motorcycle_fragments', None)
        self.assertTrue(callable(merge), 'Missing fragment reconciliation')
        boxes = [[10, 120, 130, 300], [120, 20, 250, 160]]
        result = merge(boxes, [.8, .39], [[80, 10, 170, 290]], [.9])
        self.assertEqual(result[2], [[0], [1]])

    def test_overlapping_scooter_fragments_keep_both_riders(self):
        # Rounded detections from the reported one-scooter photo.
        d = {'motorcycle': [[152, 161, 401, 450], [286, 143, 550, 446]],
             'motorcycle_score': [.781, .486],
             'person': [[425, 65, 534, 384], [301, 75, 432, 408]],
             'person_score': [.842, .829],
             'head': [[342, 76, 415, 153], [457, 59, 529, 134]],
             'head_detection': True}
        img = np.zeros((452, 678, 3), np.uint8)
        result = Pipeline(lambda _: d, lambda crops: [.1] * len(crops), n_min=1).run(
            [(0, img)], tracking=False)
        self.assertEqual([track['n_riders'] for track in result['tracks'].values()], [2])
        self.assertEqual(result['per_frame'][0][1][0][1], [152, 65, 550, 450])

        # With a separate rider supported exclusively on the left, both
        # overlapping vehicle detections have independent rider evidence.
        boxes = d['motorcycle']
        persons = [*d['person'], [172, 76, 259, 417]]
        merged = core.merge_motorcycle_fragments(boxes, d['motorcycle_score'], persons,
                                                  [*d['person_score'], .9])
        self.assertEqual(merged[0], boxes)


if __name__ == '__main__':
    unittest.main()
