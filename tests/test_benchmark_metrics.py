"""Focused, dependency-free checks for the frame benchmark contract."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from de14.metrics import evaluate_frames
import prepare_benchmark_data
from prepare_benchmark_data import select_frames, vehicles_for_frames


class FrameMetricsTests(unittest.TestCase):
    def test_real_overlap_match_and_rider_errors(self):
        frames = [{"clip": "A", "frame": 1, "vehicles": [
            {"box": [0, 0, 10, 10], "violation": True, "n_riders": 2},
            {"box": [30, 0, 40, 10], "violation": False, "n_riders": 1}]}]
        predictions = [{"clip": "A", "frame": 1, "vehicles": [
            {"box": [1, 0, 11, 10], "decision": "không đội mũ", "n_riders": 1},
            {"box": [30, 0, 40, 10], "decision": "có đội mũ", "n_riders": 1}]}]
        result = evaluate_frames(frames, predictions)
        self.assertEqual(result["vehicle_detection_precision"], 1.0)
        self.assertEqual(result["vehicle_detection_recall"], 1.0)
        self.assertEqual(result["violation_precision"], 1.0)
        self.assertEqual(result["violation_recall"], 1.0)
        self.assertEqual(result["rider_count_mae"], 0.5)
        self.assertEqual(result["rider_count_exact_rate"], 0.5)

    def test_unknown_and_missing_positive_are_false_negatives(self):
        frames = [{"clip": "A", "frame": 1, "vehicles": [
            {"box": [0, 0, 10, 10], "violation": True, "n_riders": 1}]},
            {"clip": "A", "frame": 2, "vehicles": [
                {"box": [0, 0, 10, 10], "violation": True, "n_riders": 1}]}]
        result = evaluate_frames(frames, [{"clip": "A", "frame": 1, "vehicles": [
            {"box": [0, 0, 10, 10], "decision": "chưa xác định", "n_riders": 1}]}])
        self.assertEqual(result["violation_recall"], 0.0)
        self.assertIsNone(result["violation_precision"])
        self.assertEqual(result["counts"]["violation_fn"], 2)
        self.assertEqual(result["unknown_rate"], 1.0)
        self.assertEqual(result["vehicle_detection_recall"], 0.5)

    def test_duplicate_and_unmatched_violation_predictions_are_false_positives(self):
        frames = [{"clip": "A", "frame": 1, "vehicles": [
            {"box": [0, 0, 10, 10], "violation": True, "n_riders": 1}]}]
        prediction = {"box": [0, 0, 10, 10], "decision": "không đội mũ", "n_riders": 1}
        result = evaluate_frames(frames, [{"clip": "A", "frame": 1, "vehicles": [
            prediction, dict(prediction),
            {"box": [30, 30, 40, 40], "decision": "không đội mũ", "n_riders": 1}]}])
        self.assertAlmostEqual(result["violation_precision"], 1 / 3)
        self.assertAlmostEqual(result["vehicle_detection_precision"], 1 / 3)
        self.assertEqual(result["counts"]["violation_fp"], 2)
        self.assertEqual(result["counts"]["matched_vehicles"], 1)

    def test_globally_descending_iou_matches_before_lower_overlap(self):
        frames = [{"clip": "A", "frame": 1, "vehicles": [
            {"box": [0, 0, 10, 10], "violation": False, "n_riders": 1},
            {"box": [4, 0, 14, 10], "violation": True, "n_riders": 2}]}]
        predictions = [{"clip": "A", "frame": 1, "vehicles": [
            {"box": [4, 0, 14, 10], "decision": "không đội mũ", "n_riders": 2},
            {"box": [-4, 0, 6, 10], "decision": "có đội mũ", "n_riders": 1}]}]
        result = evaluate_frames(frames, predictions)
        self.assertEqual(result["counts"]["matched_vehicles"], 2)
        self.assertEqual(result["violation_recall"], 1.0)
        self.assertEqual(result["rider_count_mae"], 0.0)

    def test_frame_identity_prevents_cross_frame_matching(self):
        frames = [{"clip": "A", "frame": 1, "vehicles": [
            {"box": [0, 0, 10, 10], "violation": True, "n_riders": 1}]},
            {"clip": "A", "frame": 2, "vehicles": []}]
        result = evaluate_frames(frames, [{"clip": "A", "frame": 2, "vehicles": [
            {"box": [0, 0, 10, 10], "decision": "không đội mũ", "n_riders": 1}]}])
        self.assertEqual(result["violation_precision"], 0.0)
        self.assertEqual(result["violation_recall"], 0.0)
        self.assertIsNone(result["rider_count_mae"])

    def test_zero_denominators_are_undefined(self):
        result = evaluate_frames([], [])
        for name in ("vehicle_detection_precision", "vehicle_detection_recall",
                     "violation_precision", "violation_recall", "unknown_rate",
                     "rider_count_mae", "rider_count_exact_rate"):
            self.assertIsNone(result[name], name)

    def test_rejects_duplicate_frame_records(self):
        frame = {"clip": "A", "frame": 1, "vehicles": []}
        with self.assertRaises(ValueError):
            evaluate_frames([frame, frame], [])
        with self.assertRaises(ValueError):
            evaluate_frames([frame], [frame, frame])

    def test_rejects_predictions_outside_manifest(self):
        with self.assertRaises(ValueError):
            evaluate_frames([], [{"clip": "A", "frame": 1, "vehicles": []}])


class BenchmarkPreparationTests(unittest.TestCase):
    def test_fresh_checkout_downloads_split_only_into_benchmark_source_and_reuses_cache(self):
        content = b"video_id,Set\nA,test\nB,training\n"
        response = SimpleNamespace(content=content, raise_for_status=lambda: None)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("prepare_benchmark_data.requests.get", return_value=response) as http:
                split_path = prepare_benchmark_data.resolve_split(root)
                self.assertEqual(split_path, root / "data/benchmark/source/data_split.csv")
                self.assertEqual(split_path.read_bytes(), content)
                self.assertFalse((root / "data/helmet").exists())
                self.assertEqual(http.call_args.args[0], prepare_benchmark_data.KAGGLE + "data_split.csv")
            with patch("prepare_benchmark_data.requests.get", side_effect=AssertionError("network on cache hit")):
                self.assertEqual(prepare_benchmark_data.resolve_split(root), split_path)

    def test_local_official_split_remains_authoritative_over_benchmark_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local = root / "data/helmet/data_split.csv"
            local.parent.mkdir(parents=True)
            local.write_bytes(b"video_id,Set\nlocal,test\n")
            cached = root / "data/benchmark/source/data_split.csv"
            cached.parent.mkdir(parents=True)
            cached.write_bytes(b"video_id,Set\nother,test\n")
            with patch("prepare_benchmark_data.requests.get", side_effect=AssertionError("network on local split")):
                self.assertEqual(prepare_benchmark_data.resolve_split(root), local)
            self.assertEqual(local.read_bytes(), b"video_id,Set\nlocal,test\n")

    def test_explicit_missing_split_fails_without_download_or_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            explicit = root / "missing.csv"
            with patch("prepare_benchmark_data.requests.get", side_effect=AssertionError("unexpected download")):
                with self.assertRaisesRegex(FileNotFoundError, "Explicit --split-csv.*missing.csv"):
                    prepare_benchmark_data.resolve_split(root, explicit)
            self.assertFalse((root / "data").exists())

    def test_explicit_existing_split_is_used_without_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            explicit = root / "supplied.csv"
            explicit.write_bytes(b"video_id,Set\nA,test\n")
            with patch("prepare_benchmark_data.requests.get", side_effect=AssertionError("unexpected download")):
                self.assertEqual(prepare_benchmark_data.resolve_split(root, explicit), explicit)

    def test_invalid_split_download_is_not_cached(self):
        response = SimpleNamespace(content=b"<html>not a CSV</html>", raise_for_status=lambda: None)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("prepare_benchmark_data.requests.get", return_value=response):
                with self.assertRaisesRegex(ValueError, "split CSV"):
                    prepare_benchmark_data.resolve_split(root)
            self.assertFalse((root / "data/benchmark/source/data_split.csv").exists())

    def test_selection_excludes_nonconsecutive_clips_and_uses_fixed_seed(self):
        grouped = {
            "testA": [(n, f"A_{n}.jpg") for n in (1, 2, 3, 4, 5, 6)],
            "testB": [(n, f"B_{n}.jpg") for n in (11, 12, 13, 14)],
            "sparse": [(n, f"S_{n}.jpg") for n in (1, 3, 5, 7)],
        }
        selected = select_frames(grouped, clips=2, frames_per_clip=4, seed=20261105)
        self.assertEqual([clip for clip, _ in selected], ["testA", "testB"])
        self.assertEqual(selected, select_frames(dict(reversed(list(grouped.items()))),
                                                clips=2, frames_per_clip=4, seed=20261105))
        for _, frames in selected:
            numbers = [number for number, _ in frames]
            self.assertEqual(numbers, list(range(numbers[0], numbers[0] + 4)))
        with self.assertRaises(ValueError):
            select_frames(grouped, clips=3, frames_per_clip=4, seed=20261105)

    def test_annotation_preserves_original_coordinates_and_all_rider_labels(self):
        annotation = ("track_id,frame_id,x,y,w,h,label\n"
                      "t1,1,100,200,30,40,DHelmetP1NoHelmetP2Helmet\n"
                      "t2,1,300,400,50,60,DHelmet\n"
                      "t1,2,110,200,30,40,DHelmetP1NoHelmetP2Helmet\n")
        result = vehicles_for_frames(annotation, [1, 3])
        self.assertEqual(result[1], [
            {"track_id": "t1", "box": [100.0, 200.0, 130.0, 240.0],
             "violation": True, "n_riders": 3},
            {"track_id": "t2", "box": [300.0, 400.0, 350.0, 460.0],
             "violation": False, "n_riders": 1}])
        self.assertEqual(result[3], [])

    def test_annotation_rejects_duplicate_track_in_a_frame(self):
        annotation = ("track_id,frame_id,x,y,w,h,label\n"
                      "t1,1,100,200,30,40,DHelmet\n"
                      "t1,1,110,200,30,40,DNoHelmet\n")
        with self.assertRaises(ValueError):
            vehicles_for_frames(annotation, [1])


if __name__ == "__main__":
    unittest.main()
