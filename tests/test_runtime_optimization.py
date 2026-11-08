"""Execution optimizations preserve head ownership, abstention and CLIP score semantics."""
import sys
import tempfile
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from de14.pipeline import Pipeline
from de14.classifiers import ClipHelmetVerifier
from de14.classifiers import VerifiedHelmetCls


class FrameBatchTest(unittest.TestCase):
    def run_scene(self, batched):
        image = np.zeros((200, 400, 3), np.uint8)
        image[10:40, 40:70] = 11
        image[55:80, 90:115] = 22
        image[10:40, 270:300] = 33
        det = lambda im: {"motorcycle": [[20, 70, 170, 190], [230, 70, 380, 190]],
                          "motorcycle_score": [.95, .95], "person": [], "head_detection": True,
                          "head": [[40, 10, 70, 40], [90, 55, 115, 80], [270, 10, 300, 40]]}
        class Classifier:
            def __init__(self): self.calls = []
            def __call__(self, crops):
                self.calls.append(len(crops))
                return [{11: .1, 22: None, 33: .9}[int(c.max())] for c in crops]
        classifier = Classifier()
        result = Pipeline(det, classifier, n_min=1, batch_heads=batched).run([(0, image)])
        return classifier.calls, result

    def test_batched_frame_preserves_unknown_and_vehicle_ownership(self):
        calls, result = self.run_scene(True)
        self.assertEqual(calls, [3])
        per_vehicle = result["per_frame"][0][1]
        self.assertEqual([p for _, p in per_vehicle[0][3]], [.1, None])
        self.assertEqual([p for _, p in per_vehicle[1][3]], [.9])
        self.assertEqual(result["classifier_calls"], 1)

    def test_baseline_and_batch_have_identical_results_with_fewer_calls(self):
        old_calls, old = self.run_scene(False)
        new_calls, new = self.run_scene(True)
        self.assertEqual(old_calls, [2, 1])
        self.assertEqual(new_calls, [3])
        self.assertEqual(old["tracks"], new["tracks"])
        self.assertEqual(old["per_frame"], new["per_frame"])


class ClipCacheTest(unittest.TestCase):
    def make_verifier(self, cached, pooled):
        class Batch(dict):
            def to(self, device): return self
        class Processor:
            def __call__(self, text=None, images=None, **kwargs):
                data = {}
                if text is not None: data["input_ids"] = torch.zeros((6, 1), dtype=torch.long)
                if images is not None: data["pixel_values"] = torch.tensor([[1., 0.]] * len(images))
                return Batch(data)
        class Model:
            logit_scale = torch.tensor(float(np.log(2)))
            def get_text_features(self, **kwargs):
                x = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.], [0., 1.], [0., 1.]])
                return SimpleNamespace(pooler_output=x) if pooled else x
            def get_image_features(self, pixel_values):
                return SimpleNamespace(pooler_output=pixel_values) if pooled else pixel_values
            def __call__(self, input_ids, pixel_values):
                return SimpleNamespace(logits_per_image=torch.tensor([[2., 2., 0., 0., 0., 0.]] * len(pixel_values)))
        verifier = ClipHelmetVerifier.__new__(ClipHelmetVerifier)
        verifier.model, verifier.proc = Model(), Processor()
        verifier.torch, verifier.device = torch, "cpu"
        verifier.cache_text, verifier._text_features = cached, None
        verifier.stats = {"text_forwards": 0, "image_forwards": 0}
        return verifier

    def test_cached_text_supports_old_and_new_transformers_feature_types(self):
        for pooled in (False, True):
            with self.subTest(pooled=pooled):
                verifier = self.make_verifier(True, pooled)
                crops = [np.zeros((40, 40, 3), np.uint8)] * 2
                self.assertEqual(verifier(crops), [-2., -2.])
                self.assertEqual(verifier(crops), [-2., -2.])
                self.assertEqual(verifier.stats, {"text_forwards": 1, "image_forwards": 2})

    def test_uncached_and_cached_margins_match(self):
        crops = [np.zeros((40, 40, 3), np.uint8)]
        old, new = self.make_verifier(False, False), self.make_verifier(True, False)
        self.assertEqual(old(crops), new(crops))
        old(crops); new(crops)
        self.assertEqual(old.stats["text_forwards"], 2)
        self.assertEqual(new.stats["text_forwards"], 1)


class RuntimeConfigurationTest(unittest.TestCase):
    def test_dino_only_experiment_preserves_size_filter_without_a_verifier(self):
        model = VerifiedHelmetCls(lambda cs: [.1, .9][:len(cs)], verifier=None)
        crops = [np.zeros((40, 40, 3), np.uint8), np.zeros((12, 12, 3), np.uint8),
                 np.zeros((40, 40, 3), np.uint8)]
        self.assertEqual(model(crops), [.1, None, .9])

    def test_yolo_helmet_classifier_obeys_the_resolved_runtime_device(self):
        from de14.runtime import RuntimeConfig, build_runtime
        class EmptyBoxes:
            def __len__(self): return 0
        seen = []
        class Model:
            names = {0: "With Helmet", 1: "Without Helmet"}
            def __init__(self, weights): pass
            def predict(self, *args, **kwargs):
                seen.append(kwargs.get("device"))
                return [SimpleNamespace(boxes=EmptyBoxes())]
        for device in ("cpu", "cuda:1"):
            with self.subTest(device=device), patch("ultralytics.YOLO", Model), \
                 patch("torch.cuda.is_available", return_value=True):
                bundle = build_runtime(RuntimeConfig(device=device, accurate=False, classifier_kind="yolo"))
                self.assertEqual(bundle.classifier([np.zeros((40, 40, 3), np.uint8)]), [None])
                self.assertEqual(seen[-1], device)

    def test_probe_model_id_stays_portable_with_a_local_cloud_checkpoint(self):
        from de14.classifiers import DinoEmbedder
        class Model:
            def to(self, device): return self
            def eval(self): return self
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/"dinov3-test").mkdir()
            with patch.dict("os.environ", {"DINO_LOCAL_DIR": directory}), \
                 patch("transformers.AutoImageProcessor.from_pretrained", return_value=object()), \
                 patch("transformers.AutoModel.from_pretrained", return_value=Model()):
                embedder = DinoEmbedder("facebook/dinov3-test", device="cpu", local_files_only=True)
                self.assertEqual(embedder.model_id, "facebook/dinov3-test")

    def test_bounded_classifier_batches_preserve_every_crop(self):
        calls = []
        def base(crops):
            calls.append(len(crops))
            return [float(c[0, 0, 0])/100 for c in crops]
        model = VerifiedHelmetCls(base, lambda cs: [-.8] * len(cs), batch_size=2)
        crops = [np.full((40, 40, 3), v, np.uint8) for v in (10, 50, 90, 20, 80)]
        self.assertEqual(model(crops), [.1, None, .9, .2, .8])
        self.assertEqual(calls, [2, 2, 1])

    def test_explicit_cuda_does_not_silently_become_cpu(self):
        from de14.runtime import resolve_device
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(RuntimeError): resolve_device("cuda")
            self.assertEqual(resolve_device("auto"), "cpu")

    def test_profile_rejects_unknown_runtime_and_negative_threads(self):
        from de14.runtime import RuntimeConfig
        with self.assertRaises(ValueError): RuntimeConfig(head_backend="unknown")
        with self.assertRaises(ValueError): RuntimeConfig(ort_threads=-1)


@unittest.skipUnless(importlib.util.find_spec("openvino"), "Optional OpenVINO runtime not installed")
class OpenVinoGatherTest(unittest.TestCase):
    def run_gather(self, data, indices, axis):
        import openvino as ov
        from openvino import opset13 as ops
        from de14.headdet import onnx_gather_elements
        d = ops.parameter(data.shape, data.dtype, name="data")
        i = ops.parameter(indices.shape, indices.dtype, name="indices")
        result = onnx_gather_elements(d, i, axis)
        compiled = ov.Core().compile_model(ov.Model([result], [d, i]), "CPU")
        return compiled([data, indices])[compiled.output(0)]

    def test_gather_allows_shorter_non_axis_dimensions(self):
        actual = self.run_gather(np.arange(12, dtype=np.float32).reshape(1, 3, 4),
                                 np.array([[[2], [0]]], np.int64), 1)
        np.testing.assert_array_equal(actual, [[[8.], [0.]]])

    def test_gather_matches_onnx_axis_zero_example(self):
        actual = self.run_gather(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], np.float32),
                                 np.array([[1, 2, 0], [2, 0, 0]], np.int32), 0)
        np.testing.assert_array_equal(actual, [[4., 8., 3.], [7., 2., 3.]])

    def test_gather_preserves_negative_axis_and_indices(self):
        actual = self.run_gather(np.array([[1, 2, 3], [4, 5, 6]], np.float32),
                                 np.array([[-1, 0], [1, -2]], np.int64), -1)
        np.testing.assert_array_equal(actual, [[3., 1.], [5., 5.]])


if __name__ == "__main__":
    unittest.main()
