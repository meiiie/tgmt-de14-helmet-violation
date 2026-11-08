"""Download the optional offline head detector + semantic hat verifier (about 810 MB).

Models stay in ignored weights/. No training and no modification of the existing DINO probe.
"""
import hashlib
import json
import os
from pathlib import Path
import urllib.request

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
HEAD_NAME = "deimv2_wholebody49_boxes_only.onnx"
HEAD_URL = ("https://github.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation/"
            "releases/download/weights/" + HEAD_NAME)
CLIP_ID = "openai/clip-vit-base-patch32"


def main():
    weights = ROOT / "weights"; weights.mkdir(exist_ok=True)
    head = weights / HEAD_NAME
    if not head.is_file() or head.stat().st_size != 202880742:
        partial = head.with_suffix(".onnx.download")
        print("Downloading DEIMv2 head detector...", flush=True)
        urllib.request.urlretrieve(HEAD_URL, partial)
        os.replace(partial, head)
    # Pin one revision so processor and weights always come from the same release.
    revision = HfApi().model_info(CLIP_ID).sha
    snapshot = snapshot_download(CLIP_ID, revision=revision,
                                 allow_patterns=["*.json", "*.txt", "*.safetensors", "pytorch_model.bin"])
    from transformers import CLIPModel, CLIPProcessor
    target = weights / "clip-vit-base-patch32"
    CLIPModel.from_pretrained(snapshot, local_files_only=True).save_pretrained(target)
    CLIPProcessor.from_pretrained(snapshot, local_files_only=True).save_pretrained(target)
    manifest = {"head_source": HEAD_URL, "head_sha256": hashlib.sha256(head.read_bytes()).hexdigest(),
                "verifier": CLIP_ID, "verifier_revision": revision,
                "head_license": "Apache-2.0", "verifier_license": "MIT",
                "note": "Optional inference models; no accuracy or calibration guarantee."}
    (weights / "accuracy_models.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("READY. Restart app.py and enable the offline accuracy checkbox.", flush=True)


if __name__ == "__main__":
    main()
