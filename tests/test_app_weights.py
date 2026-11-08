"""Kiểm tra cấu hình app khi clone repo chưa có probe, không chạy model."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app


class AppWeightsTest(unittest.TestCase):
    def test_offline_backend_loads_yolo26_checkpoint(self):
        from de14.detectors import YoloDetector
        seen = {}
        class Model:
            def __init__(self, weights): seen["weights"] = weights
        # Keep the real detector constructor and app model factory; replace only weight loading.
        with patch("ultralytics.YOLO", Model), patch("de14.classifiers.YoloHelmetCls", return_value=object()), \
             patch.object(app, "_head_ready", return_value=False):
            app._cache.clear()
            detector, _ = app.get_models(app.BACKENDS[0], 960)
            self.assertEqual(Path(seen["weights"]).resolve(), Path("weights/yolo26s.pt").resolve())
            self.assertEqual(detector.name, "yolo26s-coco")
        app._cache.clear()

    def test_default_without_probe_uses_yolo_helmet_even_with_gpu(self):
        with patch.object(app.os.path, "isfile", return_value=False), \
             patch.object(app._torch.cuda, "is_available", return_value=True):
            self.assertEqual(app._default_backend(), app.BACKENDS[0])

    def test_default_with_probe_on_cpu_keeps_dino(self):
        with patch.object(app.os.path, "isfile", return_value=True), \
             patch.object(app._torch.cuda, "is_available", return_value=False):
            self.assertEqual(app._default_backend(), app.BACKENDS[1])

    def test_default_with_probe_on_gpu_uses_offline_yolo26(self):
        with patch.object(app.os.path, "isfile", return_value=True), \
             patch.object(app._torch.cuda, "is_available", return_value=True), \
             patch.dict(app.os.environ, {"HF_TOKEN": "test-token"}):
            self.assertEqual(app._default_backend(), app.BACKENDS[1])

    def test_named_probe_is_accepted_without_renaming(self):
        with tempfile.TemporaryDirectory() as folder:
            missing = str(Path(folder) / "dino_probe.joblib")
            named = Path(folder) / "dino_probe_dinov3-vits16-pretrain-lvd1689m.joblib"
            named.touch()
            with patch.object(app, "DINO_PROBES", (missing, str(named)), create=True):
                self.assertEqual(app._require_dino_probe(), str(named))

    def test_offline_missing_probe_fails_before_loading_detector(self):
        with patch.object(app.os.path, "isfile", return_value=False), \
             patch("de14.detectors.YoloDetector") as detector:
            app._cache.clear()
            with self.assertRaises(app.gr.Error) as err:
                app.get_models(app.BACKENDS[1], 640)
            self.assertIn("dino_probe.joblib", str(err.exception))
            detector.assert_not_called()

    def test_realtime_missing_probe_reports_how_to_prepare_it(self):
        with patch.object(app.os.path, "isfile", return_value=False):
            with self.assertRaises(app.gr.Error) as err:
                app._rt_pipeline("yolo26s", 640, 5, True)
            self.assertIn("dino_probe.joblib", str(err.exception))
            self.assertIn("train_probe.py", str(err.exception))


class AppDeviceChoicesTest(unittest.TestCase):
    def setUp(self):
        app._cache.clear()
        self.addCleanup(app._cache.clear)

    def test_cpu_hides_sam_and_cuda_but_retains_available_intel_gpu(self):
        with patch.object(app._torch.cuda,'is_available',return_value=False), \
             patch.object(app,'OPENVINO_DEVICES',('CPU','GPU'),create=True):
            self.assertNotIn(app.BACKENDS[2],app._available_backends())
            profiles=dict(app._available_runtime_choices())
            self.assertNotIn('cloud_cuda',profiles.values())
            self.assertIn('openvino_gpu',profiles.values())
            self.assertIn('openvino_cpu',profiles.values())

    def test_cuda_runtime_offers_cloud_and_sam_without_requiring_openvino(self):
        with patch.object(app._torch.cuda,'is_available',return_value=True), \
             patch.object(app,'OPENVINO_DEVICES',(),create=True):
            self.assertIn(app.BACKENDS[2],app._available_backends())
            values=dict(app._available_runtime_choices()).values()
            self.assertIn('cloud_cuda',values)
            self.assertNotIn('openvino_gpu',values)

    def test_invalid_cuda_environment_default_uses_available_cpu_profile(self):
        with patch.object(app._torch.cuda,'is_available',return_value=False), \
             patch.object(app,'OPENVINO_DEVICES',('CPU',),create=True), \
             patch.dict(app.os.environ,{'HELMET_RUNTIME_PROFILE':'cloud_cuda'}):
            self.assertEqual(app._default_runtime_profile(),'openvino_cpu')

    def test_without_openvino_safe_default_is_optimized_cpu(self):
        with patch.object(app._torch.cuda,'is_available',return_value=False), \
             patch.object(app,'OPENVINO_DEVICES',(),create=True), \
             patch.dict(app.os.environ,{'HELMET_RUNTIME_PROFILE':'openvino_gpu'}):
            self.assertEqual(app._default_runtime_profile(),'optimized_cpu')

    def test_stale_cuda_or_sam_request_explains_local_choice_before_loading_models(self):
        for backend,profile in [(app.BACKENDS[2],'optimized_cpu'),(app.BACKENDS[1],'cloud_cuda')]:
            with self.subTest(backend=backend), patch.object(app._torch.cuda,'is_available',return_value=False), \
                 patch('de14.runtime.build_runtime') as factory, patch.object(app,'_require_dino_probe') as probe:
                with self.assertRaises(app.gr.Error) as error:
                    app.get_runtime(backend,1280,False,profile)
                self.assertIn('CUDA',str(error.exception))
                self.assertIn('OpenVINO',str(error.exception))
                factory.assert_not_called()
                probe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
