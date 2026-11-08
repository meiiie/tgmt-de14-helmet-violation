"""Prepare the pinned inference dependencies for a Kaggle web job."""
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


TORCH_PACKAGES = ("torch", "torchvision", "torchaudio")
ORT_PACKAGES = ("onnxruntime", "onnxruntime-gpu")


def installed_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def ort_cuda_ready(project, version):
    """Check a fresh import because the CPU and GPU wheels share a module."""
    code = """
import sys
import torch
import onnxruntime as ort
if hasattr(ort, 'preload_dlls'):
    ort.preload_dlls()
sys.exit(0 if ort.__version__ == sys.argv[1] and
         'CUDAExecutionProvider' in ort.get_available_providers() else 1)
"""
    try:
        result = subprocess.run([sys.executable, "-c", code, version], cwd=project,
                                capture_output=True, timeout=45)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def prepare_dependencies(project, meta, ort_version, run_python):
    project = Path(project)
    versions = {name: installed_version(name) for name in TORCH_PACKAGES}
    pins = {name: version for name, version in versions.items() if version}
    packages = meta["reference_packages"]
    cpu_report = project / "data/kaggle_video/openvino_cpu_report.json"
    if cpu_report.is_file():
        packages = json.loads(cpu_report.read_text(encoding="utf8"))["packages"]
    pins.update({name: packages[name] for name in ("ultralytics", "transformers", "numpy") if name in packages})
    provenance = json.loads((project / "weights/dino_probe_provenance.json").read_text(encoding="utf8"))
    if provenance.get("scikit_learn_version"):
        pins["scikit-learn"] = provenance["scikit_learn_version"]

    pending = []
    reused = []
    for line in (project / "requirements.txt").read_text(encoding="utf8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        name = canonicalize_name(requirement.name)
        if name in (*TORCH_PACKAGES, *ORT_PACKAGES):
            continue
        if requirement.marker and not requirement.marker.evaluate():
            continue
        version = installed_version(name)
        if (version and not requirement.url and not requirement.extras
                and requirement.specifier.contains(version, prereleases=True)
                and (name not in pins or version == pins[name])):
            reused.append(name)
        else:
            pending.append(line)

    if pending:
        with tempfile.TemporaryDirectory() as folder:
            constraints = Path(folder) / "constraints.txt"
            constraints.write_text("".join(f"{name}=={version}\n" for name, version in pins.items()), encoding="utf8")
            run_python("-m", "pip", "install", "-q", "-c", constraints, *pending)

    ort_reused = installed_version("onnxruntime-gpu") == ort_version and ort_cuda_ready(project, ort_version)
    if not ort_reused:
        existing = [name for name in ORT_PACKAGES if installed_version(name)]
        if existing:
            run_python("-m", "pip", "uninstall", "-y", *existing)
        run_python("-m", "pip", "install", "-q", f"onnxruntime-gpu=={ort_version}")
    return {"requirements_reused": reused, "pip_requested": pending, "ort_reused": ort_reused}
