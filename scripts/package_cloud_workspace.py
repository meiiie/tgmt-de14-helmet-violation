"""Package current source for an interactive Colab/Kaggle run, without local assets."""
from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


PROBE_FILES = ("weights/dino_probe.joblib", "weights/dino_probe_provenance.json")
BLOCKED_PARTS = {".git", ".venv", "venv", "__pycache__", "data", "datasets", "outputs"}
MAX_PROBE_BYTES = 4 * 1024 * 1024


def _allowed_source(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if any(part.lower() in BLOCKED_PARTS for part in relative.parts):
        return False
    if any(part.startswith(".") for part in relative.parts):
        return False
    if any((root.joinpath(*relative.parts[:i])).is_symlink()
           for i in range(1, len(relative.parts) + 1)):
        return False
    name = path.name.lower()
    if any(word in name for word in ("secret", "credential", ".env")):
        return False
    return path.is_file() and path.resolve().is_relative_to(root.resolve())


def package_workspace(root: Path, output: Path, include_probe: bool = False) -> list[str]:
    root = root.resolve()
    selected: set[Path] = set()
    for folder, extensions in (("src", {".py"}), ("scripts", {".py", ".sh", ".js"}),
                               ("configs", {".json"})):
        selected.update(p for p in (root / folder).rglob("*")
                        if p.suffix.lower() in extensions and _allowed_source(p, root))
    extras = [root / "README.md", root / "app.py", root / "notebooks/helmet_runtime_colab_kaggle.ipynb",
              root / "src/de14/annotation_ui.html",
              root / "docs/cloud_workflow_2026-11-05.md",
              root / "docs/crowded_rider_debug_2026-11-06.md",
              root / "docs/head_bounds_2026-11-06.md",
              root / "docs/full_length_video_2026-11-06.md",
              root / "docs/video_sampling_benchmark_2026-11-06.md",
              root / "docs/video_labeling_guide_2026-11-06.md",
              root / "docs/label_review_2026-11-06.md",
              root / "docs/video_tracking_improvements_2026-11-06.md",
              root / "docs/cuda_availability_2026-11-06.md",
              root / "docs/colab_gpu_inference_2026-11-06.md",
              root / "docs/gpu_api_web_2026-11-07.md",
              root / "docs/gpu_rental_low_cost_2026-11-07.md",
              root / "docs/kaggle_video_benchmark_2026-11-07.md",
              root / "notebooks/helmet_kaggle_video_benchmark.ipynb",
              root / "notebooks/colab_gpu_inference_cell.py",
              root / "docs/results/video_tracking_2026-11-06.json",
              root / "docs/results/video_sampling_2026-11-06.json",
              root / "docs/results/head_modes_2026-11-06.json",
              root / "docs/results/crowded_riders_2026-11-06.json",
              root / "docs/model_training_proposal_2026-11-05.md"]
    extras += list(root.glob("requirements*.txt"))
    extras += list((root / "docs").glob("runtime*.md"))
    extras += list((root / "docs").glob("benchmark*.md"))
    selected.update(p for p in extras if _allowed_source(p, root))
    if include_probe:
        for name in PROBE_FILES:
            path = root / name
            if not _allowed_source(path, root):
                raise FileNotFoundError(f"Explicit probe inclusion requires a regular file: {name}")
            if path.stat().st_size > MAX_PROBE_BYTES:
                raise ValueError(f"Refusing unexpectedly large probe artifact: {name}")
            selected.add(path)
    names = sorted(p.relative_to(root).as_posix() for p in selected)
    if not names:
        raise ValueError("No source files found")
    output = output.resolve()
    if output in (p.resolve() for p in selected):
        raise ValueError("Output must not overwrite a selected source file")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
            for name in names:
                entry = ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
                entry.create_system = 3
                entry.external_attr = 0o100644 << 16
                entry.compress_type = ZIP_DEFLATED
                archive.writestr(entry, (root / name).read_bytes())
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/cloud/helmet_workspace.zip"))
    parser.add_argument("--include-probe", action="store_true",
                        help="Explicitly include only the small DINO probe and its provenance; no model checkpoints")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.out if args.out.is_absolute() else root / args.out
    names = package_workspace(root, output, args.include_probe)
    print(f"Created {output} ({len(names)} files, {output.stat().st_size:,} bytes)")
    print("Probe included with provenance." if args.include_probe else "No weights included.")


if __name__ == "__main__":
    main()
