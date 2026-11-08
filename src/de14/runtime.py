"""One model factory and pipeline configuration for local Intel and cloud NVIDIA runs."""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def resolve_device(requested):
    import torch
    if requested == "auto": return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cpu": return requested
    if requested == "cuda" or requested.startswith("cuda:"):
        if not torch.cuda.is_available(): raise RuntimeError("CUDA requested but the current Torch/runtime has no available CUDA device")
        return requested
    raise ValueError(f"Unsupported Torch device: {requested}")


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "auto"
    imgsz: int = 1280
    cpu_threads: int = 4
    ort_threads: int = 4
    ort_spinning: bool = False
    head_backend: str = "onnx"
    openvino_device: str = "CPU"
    batch_heads: bool = True
    cache_clip_text: bool = True
    verify_with_clip: bool = True
    local_files_only: bool = True
    accurate: bool = True
    detect_heads: bool | None = None
    detector_kind: str = "yolo"
    classifier_kind: str = "dino"
    yolo_weights: str = "weights/yolo26s.pt"
    head_path: str = "weights/deimv2_wholebody49_boxes_only.onnx"
    probe_path: str = "weights/dino_probe.joblib"
    clip_path: str = "weights/clip-vit-base-patch32"

    def __post_init__(self):
        if self.head_backend not in {"onnx", "openvino"}: raise ValueError("Unknown head_backend")
        if self.detector_kind not in {"yolo", "sam3.1"}: raise ValueError("Unknown detector_kind")
        if self.classifier_kind not in {"dino", "yolo"}: raise ValueError("Unknown classifier_kind")
        if self.cpu_threads < 1 or self.ort_threads < 0: raise ValueError("Invalid thread count")
        if self.imgsz < 32: raise ValueError("Invalid detector image size")
        if self.detect_heads is not None and not isinstance(self.detect_heads, bool):
            raise ValueError("detect_heads must be a boolean or None")

    @property
    def head_enabled(self):
        # None preserves configurations written before head/CLIP were separated.
        return self.accurate if self.detect_heads is None else self.detect_heads


def load_profile(name="optimized_cpu", path=None, **overrides):
    path = Path(path) if path else ROOT / "configs/runtime_profiles.json"
    profiles = json.loads(path.read_text(encoding="utf-8"))["profiles"]
    if name not in profiles: raise ValueError(f"Unknown runtime profile: {name}")
    return replace(RuntimeConfig(**profiles[name]), **overrides)


@dataclass
class RuntimeBundle:
    detector: object
    classifier: object
    config: RuntimeConfig
    device: str

    def metadata(self):
        import torch
        head = getattr(self.detector, "hd", None)
        base = getattr(self.classifier, "base", self.classifier)
        encoder = getattr(base, "e", None)
        encoder_info = ({"model_id": encoder.model_id,
                         "revision": getattr(encoder.model.config, "_commit_hash", None),
                         "loaded_from": getattr(encoder, "loaded_from", encoder.model_id)} if encoder else None)
        active_models = [self.config.detector_kind, self.config.classifier_kind]
        if self.config.head_enabled: active_models.append("deim_head")
        if self.config.accurate and self.config.verify_with_clip: active_models.append("clip")
        return {"config": asdict(self.config), "torch_device": self.device, "active_models": active_models,
                "encoder": encoder_info,
                "head_providers": getattr(head, "providers", []),
                "cuda_device": torch.cuda.get_device_name(self.device) if self.device.startswith("cuda") else None}

    def pipeline(self, **kwargs):
        from .pipeline import Pipeline
        kwargs.setdefault('retain_untracked_heads', True)
        return Pipeline(self.detector, self.classifier, batch_heads=self.config.batch_heads, **kwargs)


def build_runtime(config, reuse=None):
    import torch
    from .detectors import YoloDetector, Sam3Detector, HeadEnhancedDetector
    from .headdet import DeimHeadDetector
    from .classifiers import DinoProbe, YoloHelmetCls, ClipHelmetVerifier, VerifiedHelmetCls
    device = resolve_device(config.device)
    if config.head_backend == "openvino" and device.startswith("cuda"):
        raise ValueError("Use separate explicit Intel OpenVINO and NVIDIA CUDA profiles")
    torch.set_num_threads(config.cpu_threads)
    compatible = (reuse is not None and reuse.device == device and config.head_enabled == reuse.config.head_enabled
                  and replace(config, accurate=reuse.config.accurate, verify_with_clip=reuse.config.verify_with_clip) == reuse.config)
    if compatible:
        detector = reuse.detector
        classifier = getattr(reuse.classifier, "base", reuse.classifier)
    else:
        detector = (Sam3Detector(device=device) if config.detector_kind == "sam3.1" else
                    YoloDetector(weights=str(ROOT / config.yolo_weights), imgsz=config.imgsz, conf=.2, device=device))
        classifier = (DinoProbe.load(str(ROOT / config.probe_path), device=device, local_files_only=config.local_files_only) if config.classifier_kind == "dino" else
                      YoloHelmetCls(weights=str(ROOT / "weights/helmet_yolo11s_nnsohamnn.pt"), device=device))
        if config.head_enabled:
            head = DeimHeadDetector(str(ROOT / config.head_path), device=device, min_score=.3,
                                    require_gpu=device.startswith("cuda"), threads=config.ort_threads,
                                    spinning=config.ort_spinning, backend=config.head_backend, openvino_device=config.openvino_device)
            detector = HeadEnhancedDetector(detector, head)
    if config.head_enabled or config.accurate:
        verifier = None
        if config.accurate and config.verify_with_clip:
            verifier = getattr(reuse.classifier, "verifier", None) if compatible else None
            if verifier is None:
                verifier = ClipHelmetVerifier(str(ROOT / config.clip_path), device=device, cache_text=config.cache_clip_text)
        classifier = VerifiedHelmetCls(classifier, verifier)
    return RuntimeBundle(detector, classifier, config, device)
