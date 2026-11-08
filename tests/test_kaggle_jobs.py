"""Exercise Kaggle job staging and versioned output selection without spending GPU quota."""
import json
from io import BytesIO
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from de14.kaggle_jobs import KaggleBatch
from scripts import kaggle_runtime_setup


class KaggleBatchTest(unittest.TestCase):
    def test_private_batch_uses_exact_run_version_and_cleans_input(self):
        job_id = "a" * 32
        notebook = "test-user/helmet-notebook"
        models = "test-user/helmet-models"
        dino = "test-user/private-dino"
        dataset = f"test-user/helmet-job-{job_id}"
        expected_run = f"{notebook}/7"
        expected_result = f"helmet-result-{job_id}.zip"
        states = iter(["QUEUED", "RUNNING", "COMPLETE"])
        calls = []
        now = 100.0

        def sleep(seconds):
            nonlocal now
            now += seconds

        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {
            "HELMET_KAGGLE_NOTEBOOK": notebook, "HELMET_KAGGLE_MODELS": models,
            "HELMET_KAGGLE_CLI": "kaggle", "HELMET_KAGGLE_DINO": dino,
        }), patch("de14.gpu_api.lock_storage"), patch("de14.gpu_api.unlock_storage"):
            work = Path(temp)
            request = work / "request.zip"
            request.write_bytes(b"test request")

            def cli(*args, **kwargs):
                nonlocal now
                now += 1
                calls.append(args)
                if args[:2] == ("datasets", "create"):
                    self.assertNotIn("--public", args)
                    uploaded = list((work / "input").glob("*.bin"))
                    self.assertEqual([file.name for file in uploaded], [f"helmet-job-{job_id}.bin"])
                    with ZipFile(uploaded[0]) as bundle:
                        self.assertEqual(bundle.read("request.zip"), b"test request")
                        with ZipFile(BytesIO(bundle.read("source.zip"))) as source:
                            self.assertIn("scripts/run_gpu_job.py", source.namelist())
                            self.assertNotIn("weights/dino_probe.joblib", source.namelist())
                    return "Your private Dataset is being created."
                if args[:2] == ("kernels", "push"):
                    metadata = json.loads((work / "kernel/kernel-metadata.json").read_text(encoding="utf8"))
                    self.assertEqual(metadata["dataset_sources"], [models, dataset, dino])
                    self.assertIs(metadata["is_private"], True)
                    self.assertIn("NvidiaTeslaT4", args)
                    return "Kernel version 7 successfully pushed."
                if args[:2] == ("kernels", "status"):
                    status = "ERROR" if args[2] == notebook else next(states)
                    return f'{args[2]} has status "KernelWorkerStatus.{status}"'
                if args[:2] == ("datasets", "status"):
                    return "ready"
                if args[:2] == ("kernels", "output"):
                    self.assertEqual(args[2], expected_run)
                    (work / expected_result).write_bytes(b"test result")
                    (work / f"helmet-timing-{job_id}.json").write_text(json.dumps({
                        "job_id": job_id, "inputs_seconds": 1, "cuda_check_seconds": 2, "dependencies_seconds": 3,
                    }), encoding="utf8")
                    return ""
                if args[:2] == ("datasets", "delete"):
                    self.assertEqual(args[-1], dataset)
                    return ""
                self.fail(f"Unexpected CLI command {args}")

            batch = KaggleBatch()
            with patch.object(batch, "_run", side_effect=cli), \
                    patch("de14.kaggle_jobs.time.sleep", side_effect=sleep), \
                    patch("de14.kaggle_jobs.time.monotonic", side_effect=lambda: now), \
                    patch("de14.kaggle_jobs.JOB_STORAGE", work / "storage"):
                result, timing = batch.run(request, job_id, work, lambda *_args, **_kwargs: None)
            self.assertEqual(result.read_bytes(), b"test result")
            self.assertEqual(calls[-1], ("datasets", "delete", "-y", dataset))
            self.assertEqual(timing["upload_seconds"], 1)
            self.assertEqual(timing["queue_wait_observed_seconds"], 12)
            self.assertEqual(timing["remote_run_observed_seconds"], 11)
            self.assertEqual(timing["environment"]["dependencies_seconds"], 3)
            self.assertTrue((work / "storage" / "timings" / f"{job_id}.json").is_file())

    def test_dependency_reuse_checks_pins_and_actual_ort_provider(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            (project / "weights").mkdir()
            (project / "weights/dino_probe_provenance.json").write_text('{"scikit_learn_version":"1.9.1"}')
            (project / "requirements.txt").write_text("numpy\nscikit-learn\nav>=19,<20\nonnxruntime\n")
            versions = {"torch": "2.11.0+cu130", "numpy": "2.5.4", "scikit-learn": "1.9.1",
                        "av": "19.0.2", "onnxruntime-gpu": "1.30.0"}
            meta = {"reference_packages": {"numpy": "2.5.4"}}
            commands = []
            run = lambda *args: commands.append(args)
            with patch.object(kaggle_runtime_setup, "installed_version", side_effect=versions.get), \
                    patch.object(kaggle_runtime_setup, "ort_cuda_ready", return_value=True):
                report = kaggle_runtime_setup.prepare_dependencies(project, meta, "1.30.0", run)
                self.assertEqual(commands, [])
                self.assertTrue(report["ort_reused"])
                self.assertEqual(set(report["requirements_reused"]), {"numpy", "scikit-learn", "av"})

            versions["numpy"] = "2.4.0"
            with patch.object(kaggle_runtime_setup, "installed_version", side_effect=versions.get), \
                    patch.object(kaggle_runtime_setup, "ort_cuda_ready", return_value=False):
                report = kaggle_runtime_setup.prepare_dependencies(project, meta, "1.30.0", run)
                self.assertEqual(report["pip_requested"], ["numpy"])
                self.assertFalse(report["ort_reused"])
                self.assertEqual(commands[-2], ("-m", "pip", "uninstall", "-y", "onnxruntime-gpu"))
                self.assertEqual(commands[-1], ("-m", "pip", "install", "-q", "onnxruntime-gpu==1.30.0"))


if __name__ == "__main__":
    unittest.main()
