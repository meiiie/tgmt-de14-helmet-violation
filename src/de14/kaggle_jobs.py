"""Submit one private Kaggle notebook run for a portable inference job."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[2]
JOB_STORAGE = ROOT / "outputs/kaggle_web"
SLUG = re.compile(r"[a-z0-9-]+/[a-z0-9-]+")
ACTIVE = {"QUEUED", "RUNNING", "PENDING", "INITIALIZING"}
FAILED = {"ERROR", "FAILED", "CANCELED", "CANCELLED", "CANCEL_ACKNOWLEDGED"}


def _write_kernel(directory, notebook, models, dataset, job_id, source_zip, dino_dataset=None):
    import runpy
    import inspect

    cells = json.loads((ROOT / "notebooks/helmet_kaggle_video_benchmark.ipynb").read_text(encoding="utf8"))["cells"]
    code = {cell["id"]: "".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"}
    source_updater = runpy.run_path(str(ROOT / "notebooks/colab_gpu_inference_cell.py"))["update_project_sources"]
    with ZipFile(source_zip) as archive:
        requirements = archive.read("requirements.txt").decode("utf8")
    setup = ("import time\nphase_started = time.perf_counter()\n"
             f"JOB_ID = {job_id!r}\nREQUIREMENTS = {requirements!r}\n") + inspect.getsource(source_updater)
    setup += r"""
import hashlib, json, os, shutil, stat, subprocess, sys, tempfile
from pathlib import Path
from zipfile import ZipFile

BASE = Path('/kaggle/working')
PROJECT = Path(tempfile.mkdtemp(prefix='helmet_job_'))

def find_one(name, required=True):
    found = list(Path('/kaggle/input').rglob(name))
    if len(found) != 1 and (found or required):
        raise RuntimeError(f'Expected exactly one private input file: {name}')
    return found[0] if found else None

def run_python(*args):
    subprocess.run([sys.executable, *map(str, args)], cwd=PROJECT, check=True)

dino_meta = find_one('dinov3-private-manifest.json', required=False)
if dino_meta:
    snapshot = json.loads(dino_meta.read_text(encoding='utf8'))
    model_name = 'dinov3-vits16-pretrain-lvd1689m'
    if snapshot['model_id'] != 'facebook/' + model_name:
        raise RuntimeError('Unexpected private DINOv3 model.')
    local = PROJECT / 'checkpoints' / model_name
    local.mkdir(parents=True)
    for name in ('config.json', 'model.safetensors', 'preprocessor_config.json'):
        target = local / name
        shutil.copy2(dino_meta.parent / f'dinov3-{name}', target)
        with target.open('rb') as file:
            if hashlib.file_digest(file, 'sha256').hexdigest() != snapshot['files'][name]:
                raise RuntimeError(f'DINOv3 checksum differs: {name}')
    os.environ['DINO_LOCAL_DIR'] = str(local.parent)
    print('Private DINOv3 revision:', snapshot['revision'], flush=True)
else:
    from kaggle_secrets import UserSecretsClient
    try:
        os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')
    except Exception:
        raise RuntimeError('HF_TOKEN is unavailable. Stage a private DINOv3 checkpoint or attach the secret.') from None
    if not os.environ['HF_TOKEN']:
        raise RuntimeError('HF_TOKEN is empty.')

job_dir = PROJECT / 'job'
job_dir.mkdir()
with ZipFile(find_one(f'helmet-job-{JOB_ID}.bin')) as job:
    if sorted(job.namelist()) != ['request.zip', 'source.zip']:
        raise RuntimeError('Unexpected Kaggle job bundle contents.')
    for name in ('request.zip', 'source.zip'):
        with job.open(name) as source, (job_dir / name).open('wb') as target:
            shutil.copyfileobj(source, target)
update_project_sources(job_dir / 'source.zip', PROJECT)
(PROJECT/'requirements.txt').write_text(REQUIREMENTS, encoding='utf8')
model_zip = find_one('helmet_kaggle_input.zip', required=False)
model_meta = None if model_zip else find_one('kaggle_input.json')
assets = (
    'weights/yolo26s.pt', 'weights/deimv2_wholebody49_boxes_only.onnx',
    'weights/dino_probe.joblib', 'weights/dino_probe_provenance.json',
)
if model_zip:
    with ZipFile(model_zip) as models:
        meta = json.loads(models.read('kaggle_input.json'))
        for name in assets:
            target = PROJECT/name
            target.parent.mkdir(parents=True, exist_ok=True)
            with models.open(name) as source, target.open('wb') as file:
                shutil.copyfileobj(source, file)
else:
    meta = json.loads(model_meta.read_text(encoding='utf8'))
    for name in assets:
        target = PROJECT/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_meta.parent/name, target)
for name in assets:
    with (PROJECT/name).open('rb') as file:
        if hashlib.file_digest(file, 'sha256').hexdigest() != meta['files'][name]:
            raise RuntimeError(f'Model checksum differs: {name}')
timing = {'job_id': JOB_ID, 'inputs_seconds': time.perf_counter() - phase_started}
"""
    execute = r"""
import onnxruntime as ort
if hasattr(ort, 'preload_dlls'):
    ort.preload_dlls()
if 'CUDAExecutionProvider' not in ort.get_available_providers():
    raise RuntimeError('CUDAExecutionProvider is unavailable.')
from scripts.run_gpu_job import execute_job
archive = execute_job(REQUEST, PROJECT/'run')
shutil.copy2(archive, BASE/f'helmet-result-{JOB_ID}.zip')
"""
    # A new interpreter is required after replacing the ONNX Runtime wheel.
    runner = ("import os, shutil, torch\nfrom pathlib import Path\n"
              "PROJECT=Path.cwd()\nBASE=Path('/kaggle/working')\n"
              f"JOB_ID={job_id!r}\nREQUEST=PROJECT/'job/request.zip'\n" + execute)
    compile(runner, "kaggle_job_runner", "exec")
    finish = f"run_python('-u', '-c', {runner!r})\n"
    gpu_check = ("phase_started = time.perf_counter()\n" + code["kg-video-04"] +
                 "\ntiming['cuda_check_seconds'] = time.perf_counter() - phase_started\n")
    install = """
import runpy
phase_started = time.perf_counter()
prepare = runpy.run_path(str(PROJECT / 'scripts/kaggle_runtime_setup.py'))['prepare_dependencies']
timing.update(prepare(PROJECT, meta, ORT_VERSION, run_python))
timing['dependencies_seconds'] = time.perf_counter() - phase_started
print('HELMET_ENVIRONMENT_TIMING=' + json.dumps(timing), flush=True)
(BASE / f'helmet-timing-{JOB_ID}.json').write_text(json.dumps(timing), encoding='utf8')
"""
    sections = [
        ("Chuẩn bị dữ liệu và trọng số", setup), ("Kiểm tra GPU", gpu_check),
        ("Kiểm tra và cài thư viện cần thiết", install), ("Nhận diện và xuất kết quả", finish),
    ]
    kernel_cells = []
    for index, (title, source) in enumerate(sections):
        compile(source, f"kaggle_job_cell_{index}", "exec")
        kernel_cells.append({"id": f"section-{index}", "cell_type": "markdown", "metadata": {},
                             "source": [f"## {title}\n"]})
        kernel_cells.append({"id": f"job-{index}", "cell_type": "code", "execution_count": None,
                             "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)})
    (directory / "run.ipynb").write_text(json.dumps({
        "nbformat": 4, "nbformat_minor": 5, "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        }, "cells": kernel_cells
    }), encoding="utf8")
    datasets = [models, dataset]
    if dino_dataset:
        datasets.append(dino_dataset)
    (directory / "kernel-metadata.json").write_text(json.dumps({
        "id": notebook, "title": notebook.split("/")[1].replace("-", " ").title(),
        "code_file": "run.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": datasets,
    }), encoding="utf8")


def _environment_timing(path, job_id):
    try:
        result = json.loads(path.read_text(encoding="utf8"))
    except (OSError, ValueError):
        return None
    if not isinstance(result, dict) or result.get("job_id") != job_id:
        return None
    for key in ("inputs_seconds", "cuda_check_seconds", "dependencies_seconds"):
        value = result.get(key)
        if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
            return None
    return result


class KaggleBatch:
    def __init__(self):
        self.notebook = os.environ.get("HELMET_KAGGLE_NOTEBOOK", "").strip()
        self.models = os.environ.get("HELMET_KAGGLE_MODELS", "").strip()
        self.dino_dataset = os.environ.get("HELMET_KAGGLE_DINO", "").strip()
        self.cli = os.environ.get("HELMET_KAGGLE_CLI") or shutil.which("kaggle")
        if not SLUG.fullmatch(self.notebook) or not SLUG.fullmatch(self.models) or not self.cli:
            raise ValueError("Thiếu cấu hình Kaggle trên máy chạy web.")
        if self.dino_dataset and not SLUG.fullmatch(self.dino_dataset):
            raise ValueError("Tên dataset DINOv3 Kaggle không hợp lệ.")

    def _run(self, *args, timeout=45):
        try:
            result = subprocess.run([self.cli, *map(str, args)], capture_output=True,
                                    encoding="utf8", errors="replace", timeout=timeout)
        except subprocess.TimeoutExpired as error:
            raise TimeoutError(f"Kaggle CLI quá thời gian ở bước {' '.join(args[:2])}.") from error
        output = result.stdout + result.stderr
        if result.returncode != 0:
            hint = " Kiểm tra quyền hoặc đăng nhập CLI." if re.search(r"\b(401|403|404)\b", output) else ""
            raise RuntimeError(f"Kaggle CLI lỗi ở bước {' '.join(args[:2])}.{hint}")
        return output.strip()

    def _status(self, run):
        output = self._run("kernels", "status", run)
        match = re.search(r'has status "(?:\w+\.)?([A-Z_]+)"', output)
        if not match:
            raise RuntimeError("Không đọc được trạng thái notebook Kaggle.")
        return match[1]

    def _wait_dataset(self, dataset):
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            state = self._run("datasets", "status", dataset).lower()
            if state == "ready":
                return
            if state in {"error", "failed"}:
                raise RuntimeError("Kaggle không chuẩn bị được dữ liệu cho job.")
            time.sleep(5)
        raise TimeoutError("Kaggle chưa chuẩn bị xong dữ liệu sau 5 phút.")

    def run(self, archive, job_id, work, progress):
        from .gpu_api import lock_storage, unlock_storage
        import runpy
        package = runpy.run_path(str(ROOT / "scripts/package_cloud_workspace.py"))["package_workspace"]

        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise ValueError("Job ID không hợp lệ.")
        JOB_STORAGE.mkdir(parents=True, exist_ok=True)
        try:
            lock = lock_storage(JOB_STORAGE)
        except RuntimeError as error:
            raise RuntimeError("Web đang chờ kết quả của một job Kaggle khác.") from error
        timing_dir = JOB_STORAGE / "timings"
        started = time.monotonic()
        timing = {"job_id": job_id, "status": "preparing", "queue_wait_observed_seconds": None,
                  "remote_run_observed_seconds": None}
        dataset = f"{self.notebook.split('/')[0]}/helmet-job-{job_id}"
        submitted = False
        created = False
        terminal = False
        try:
            if self._status(self.notebook) in ACTIVE:
                raise RuntimeError("Notebook Kaggle đang xử lý job trước.")
            if self._run("datasets", "status", self.models).lower() != "ready":
                raise RuntimeError("Bộ trọng số Kaggle chưa sẵn sàng.")
            if self.dino_dataset and self._run("datasets", "status", self.dino_dataset).lower() != "ready":
                raise RuntimeError("Bộ trọng số DINOv3 riêng tư chưa sẵn sàng.")
            data = work / "input"
            kernel = work / "kernel"
            data.mkdir()
            kernel.mkdir()
            source = work / "source.zip"
            package(ROOT, source)
            with ZipFile(data / f"helmet-job-{job_id}.bin", "w") as bundle:
                bundle.write(source, "source.zip")
                bundle.write(archive, "request.zip")
            (data / "dataset-metadata.json").write_text(json.dumps({
                "id": dataset, "title": f"Helmet job {job_id}",
                "licenses": [{"name": "other"}],
                "description": "Private temporary inference input. Rights remain with the original owners.",
            }), encoding="utf8")
            _write_kernel(kernel, self.notebook, self.models, dataset, job_id, source, self.dino_dataset)
            timing["local_prepare_seconds"] = time.monotonic() - started
            progress(None, desc="Đang tải file lên Kaggle")
            phase_started = time.monotonic()
            try:
                created_message = self._run("datasets", "create", "-p", data, "--keep-tabular", "--quiet", timeout=600)
            finally:
                timing["upload_seconds"] = time.monotonic() - phase_started
            if "Your private Dataset is being created" not in created_message:
                raise RuntimeError("Kaggle từ chối tạo dataset riêng tư cho job.")
            created = True
            phase_started = time.monotonic()
            self._wait_dataset(dataset)
            timing["dataset_ready_seconds"] = time.monotonic() - phase_started
            submitted = True  # Preserve the input if submission status is ambiguous.
            phase_started = time.monotonic()
            pushed = self._run("kernels", "push", "-p", kernel, "--accelerator", "NvidiaTeslaT4", "--timeout", "3600")
            submitted_at = time.monotonic()
            timing["submit_seconds"] = submitted_at - phase_started
            version = re.search(r"Kernel version (\d+) successfully pushed", pushed)
            if not version:
                raise RuntimeError("Kaggle chưa xác nhận lượt chạy. Kiểm tra notebook.")
            run = f"{self.notebook}/{version[1]}"
            timing["run"] = run
            print(f"Kaggle GPU run: {run}", flush=True)
            deadline = submitted_at + 3600
            running_at = None
            while time.monotonic() < deadline:
                state = self._status(run)
                observed_at = time.monotonic()
                timing["status"] = state
                if state == "COMPLETE":
                    terminal = True
                    if running_at is not None:
                        timing["remote_run_observed_seconds"] = observed_at - running_at
                    break
                if state in FAILED:
                    terminal = True
                    logs = self._run("kernels", "logs", run)
                    detail = (" Kaggle không đọc được HF_TOKEN. Cấu hình dataset DINOv3 riêng tư."
                              if "HF_TOKEN is unavailable" in logs or "HF_TOKEN is empty" in logs else "")
                    raise RuntimeError(f"Job Kaggle {run} dừng với trạng thái {state}.{detail}")
                if state == "RUNNING":
                    if running_at is None:
                        running_at = observed_at
                        timing["queue_wait_observed_seconds"] = running_at - submitted_at
                    progress(None, desc=f"Kaggle đang chạy phiên {version[1]} · đã chạy {observed_at-running_at:.0f}s")
                else:
                    progress(None, desc=f"Kaggle chờ cấp GPU · phiên {version[1]} · đã chờ "
                                        f"{observed_at-submitted_at:.0f}s · tải lên {timing['upload_seconds']:.0f}s")
                time.sleep(10)
            else:
                raise TimeoutError(f"Đã chờ một giờ. Theo dõi phiên Kaggle {run}.")
            progress(None, desc="Đang tải kết quả từ Kaggle")
            target = f"helmet-result-{job_id}.zip"
            environment_file = f"helmet-timing-{job_id}.json"
            phase_started = time.monotonic()
            self._run("kernels", "output", run, "-p", work, "--file-pattern",
                      f"^(?:{re.escape(target)}|{re.escape(environment_file)})$", "--quiet", timeout=600)
            timing["download_seconds"] = time.monotonic() - phase_started
            result = work / target
            if not result.is_file():
                raise RuntimeError(f"Kaggle chưa trả file kết quả của phiên {run}.")
            timing["environment"] = _environment_timing(work / environment_file, job_id)
            return result, timing
        finally:
            try:
                if created and (terminal or not submitted):
                    try:
                        self._run("datasets", "delete", "-y", dataset)
                    except (RuntimeError, OSError, TimeoutError):
                        print(f"Kaggle dataset cleanup pending: {dataset}", flush=True)
            finally:
                unlock_storage(lock)
                timing["job_seconds"] = time.monotonic() - started
                try:
                    timing_dir.mkdir(parents=True, exist_ok=True)
                    (timing_dir / f"{job_id}.json").write_text(
                        json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf8")
                except OSError:
                    print(f"Kaggle timing report could not be saved: {job_id}", flush=True)
