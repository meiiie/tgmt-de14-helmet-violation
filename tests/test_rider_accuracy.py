"""Regression checks for occluded riders and unsupported helmet decisions (no model downloads)."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from de14.core import decide, OK, UNDET, VIOL
from de14.pipeline import Pipeline, heads_for_riders


class RiderAccuracyTest(unittest.TestCase):
    def run_frame(self, people, heads, probabilities, head_detection=True):
        detections = {"motorcycle": [[20, 70, 180, 190]], "motorcycle_score": [.9],
                      "person": people, "head": heads, "head_detection": head_detection}
        detector = lambda image: detections
        classifier = lambda crops: probabilities[:len(crops)]
        return Pipeline(detector, classifier, n_min=1).run([(0, np.zeros((200, 200, 3), np.uint8))])

    def test_head_is_not_reused_for_two_overlapping_people(self):
        people = [[20, 10, 100, 180], [30, 10, 110, 180]]
        heads = heads_for_riders(people, [[40, 10, 70, 40]])
        self.assertNotEqual(heads[0], heads[1])

    def test_child_head_is_counted_without_a_full_body_detection(self):
        result = self.run_frame([[30, 10, 120, 180]],
                                [[40, 10, 70, 40], [100, 60, 125, 85]], [.1, .9])
        self.assertEqual(len(result["per_frame"][0][1][0][3]), 2)
        self.assertEqual(next(iter(result["tracks"].values()))["decision"], VIOL)

    def test_duplicate_body_boxes_do_not_create_extra_riders(self):
        result = self.run_frame([[30, 10, 120, 180], [32, 12, 122, 181]],
                                [[40, 10, 70, 40]], [.1, .1])
        self.assertEqual(len(result["per_frame"][0][1][0][3]), 1)

    def test_motorcycle_without_head_or_person_is_not_a_helmeted_rider(self):
        result = self.run_frame([], [], [.05])
        track = next(iter(result["tracks"].values()))
        self.assertEqual(track["decision"], UNDET)

    def test_one_unclassified_rider_prevents_an_all_helmeted_decision(self):
        result = self.run_frame([[30, 10, 120, 180]],
                                [[40, 10, 70, 40], [100, 60, 125, 85]], [.05, None])
        self.assertEqual(next(iter(result["tracks"].values()))["decision"], UNDET)

    def test_stationary_head_only_rider_survives_default_video_filter(self):
        detector = lambda im: {"motorcycle": [[20, 70, 180, 190]], "motorcycle_score": [.9],
                               "person": [], "head": [[40, 10, 70, 40]], "head_detection": True}
        result = Pipeline(detector, lambda cs: [.9] * len(cs), n_min=3,
                          track_filter={"tau_disp": .75, "tau_person_static": .7}).run(
            [(i, np.zeros((200, 200, 3), np.uint8)) for i in range(3)])
        self.assertEqual(len(result["tracks"]), 1)
        self.assertEqual(next(iter(result["tracks"].values()))["decision"], VIOL)

    def test_borderline_probabilities_do_not_become_confident_votes(self):
        self.assertEqual(decide([.49, .51, .48], n_min=1)[0], UNDET)
        self.assertEqual(decide([.1, .2, .3])[0], OK)
        self.assertEqual(decide([.8, .9, .7])[0], VIOL)

    def test_csv_includes_detected_rider_count(self):
        import csv
        detector = lambda image: {"motorcycle": [[20, 70, 180, 190]], "motorcycle_score": [.9],
                                  "person": [], "head": [[40, 10, 70, 40], [100, 60, 125, 85]],
                                  "head_detection": True}
        with tempfile.TemporaryDirectory() as folder:
            Pipeline(detector, lambda crops: [.1] * len(crops), n_min=1).run(
                [(0, np.zeros((200, 200, 3), np.uint8))], out_dir=folder, write_video=False)
            with open(Path(folder) / "video_tracks.csv", encoding="utf-8") as f:
                self.assertEqual(next(csv.DictReader(f)).get("n_riders"), "2")


class HelmetEvidenceTest(unittest.TestCase):
    def test_low_confidence_helmet_detection_does_not_invert_into_a_violation(self):
        from de14.classifiers import YoloHelmetCls
        class Boxes:
            conf = np.array([.3]); cls = np.array([0])
            def __len__(self): return 1
        boxes = Boxes()
        model = SimpleNamespace(names={0: "With Helmet", 1: "Without Helmet"},
                                predict=lambda *args, **kw: [SimpleNamespace(boxes=boxes)])
        with patch("ultralytics.YOLO", return_value=model):
            self.assertEqual(YoloHelmetCls()([np.zeros((40, 40, 3), np.uint8)]), [None])

    def test_fur_hat_semantic_evidence_overrules_a_helmet_prediction(self):
        from de14.classifiers import VerifiedHelmetCls
        model = VerifiedHelmetCls(lambda cs: [.1] * len(cs), lambda cs: [3.2] * len(cs))
        self.assertGreater(model([np.zeros((50, 50, 3), np.uint8)])[0], .8)

    def test_ambiguous_hat_is_unknown_instead_of_helmeted(self):
        from de14.classifiers import VerifiedHelmetCls
        model = VerifiedHelmetCls(lambda cs: [.1] * len(cs), lambda cs: [.2] * len(cs))
        self.assertEqual(model([np.zeros((50, 50, 3), np.uint8)]), [None])

    def test_tiny_head_is_counted_but_not_classified_from_upsampled_pixels(self):
        from de14.classifiers import VerifiedHelmetCls
        model = VerifiedHelmetCls(lambda cs: [.1] * len(cs), lambda cs: [-3.] * len(cs))
        self.assertEqual(model([np.zeros((15, 18, 3), np.uint8)]), [None])

    def test_strongly_helmeted_head_keeps_its_label(self):
        from de14.classifiers import VerifiedHelmetCls
        model = VerifiedHelmetCls(lambda cs: [.05] * len(cs), lambda cs: [-3.] * len(cs))
        self.assertEqual(model([np.zeros((50, 50, 3), np.uint8)]), [.05])


class HeadDetectionTest(unittest.TestCase):
    def test_head_enhancement_removes_duplicate_motorcycles_without_losing_heads(self):
        from de14.detectors import HeadEnhancedDetector
        detector = lambda im: {"motorcycle": [[10, 50, 90, 150], [12, 52, 92, 152]],
                               "motorcycle_score": [.9, .5], "person": []}
        head = [30, 10, 50, 30]
        result = HeadEnhancedDetector(detector, lambda im: [(.8, head)])(np.zeros((200, 200, 3)))
        self.assertEqual(len(result["motorcycle"]), 1)
        self.assertEqual(result["head"], [head])


if __name__ == "__main__":
    unittest.main()
