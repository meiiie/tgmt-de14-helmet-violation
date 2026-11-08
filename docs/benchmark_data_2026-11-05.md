# HELMET official-test smoke subset — 2026-11-05

Prepared actual data at `data/benchmark/manifest.json`: **4 clips, 16 frames,
32 annotated vehicle observations, 8 violation observations**. This subset is
for quick correctness/performance comparisons. It cannot establish representative
accuracy, and the 16 observations are correlated within their four clips.

| Official test clip | Consecutive frame IDs | GT vehicle observations | GT violations |
| --- | --- | ---: | ---: |
| Bago_rural_15 | 58–61 | 8 | 4 |
| Bago_rural_20 | 27–30 | 4 | 4 |
| Mandalay_1_223 | 38–41 | 8 | 0 |
| Mandalay_1_71 | 24–27 | 12 | 0 |
| Total | 16 frames | 32 | 8 |

`data/helmet/data_split.csv` is the authoritative existing split and was read
without changes. Its actual `Set` values are `training` (636 clips),
`validation` (92), and **`test` (182)**. The manifest normalizes `test` to
`split: "testing"`; it does not mistake validation for testing. Existing
`train_clips.txt`, training frames, probe data, and training annotations were
not written. No retraining occurs.

For a fresh cloud checkout without `data/helmet/data_split.csv`, the preparer
downloads the public Kaggle `data_split.csv` into
`data/benchmark/source/data_split.csv` and reuses that cache on later runs.
It never creates or writes the training data directory for this fallback.
An existing local official split takes precedence over the benchmark cache.
`--split-csv PATH` explicitly selects another authoritative CSV; a missing supplied
path fails clearly and does not silently download or fall back to another file.
Split source URL and SHA-256 remain recorded in the manifest.

Selection uses `random.Random(20261105)`, a sorted list of official-test clips
available in the HF mirror, and only clips containing a four-frame consecutive
window. It samples four clips, then one available consecutive window per clip.
The cached metadata exposes 80 official-test clips with images. Neither predicted
outcomes nor annotation labels enter selection. These four clips were not picked
for their number of violations.

## Sources and artifacts

- [Kaggle annotation/split mirror](https://www.kaggle.com/datasets/hozngvan/helmet-detection),
  with API base `https://www.kaggle.com/api/v1/datasets/download/hozngvan/helmet-detection/`.
  The preparer reuses an existing annotation CSV read-only when present; all four
  selected annotations were absent in the existing cache and were downloaded into
  `data/benchmark/annotations/`.
- [HF frame mirror](https://huggingface.co/datasets/eiglesiasr/nohelmetnet), metadata
  [API](https://huggingface.co/api/datasets/eiglesiasr/nohelmetnet), repository
  revision `c304ab9a380c8b88a673e913e4b105040a21db96`. The preparer caches the metadata
  in `data/benchmark/hf_metadata.json` to preserve the selection population on rerun.
- All 16 JPEGs were checked as **640×640**. Annotation boxes remain original
  **1920×1080 xyxy**. Consumers must resize each mirror image to **1920×1080 before
  inference**, then evaluate predictions in that coordinate system. Upscaling
  cannot recover the original image detail.
- All 22 source files (split, metadata, 4 annotation CSVs, 16 JPEGs) have source URLs
  and SHA-256 values in the manifest; each frame also carries its image SHA-256.
  Every file hash was checked against the generated manifest.
- `data/` is already ignored by Git. The preparer and this document are versioned;
  downloaded benchmark artifacts remain local.

Recorded SHA-256 values:

```text
manifest.json       3134f196ed85fb35f97f87fe1e8b839f947993ed271aa4794a6086027f83ea59
data_split.csv      cedcbf8b5b05eb6ec8372e395b74b9192e013854e5edd3204ac76989172e37a1
hf_metadata.json    c17300b8856c075c39a979acb7958d8d5ff7a8000106cc701feaddbc46ee5971
```

The URLs using `main` and the public mirrors can change; preserve the manifest,
cached metadata and recorded hashes when comparing runs. GT can be incomplete:
an unmatched predicted vehicle violation is scored as FP against available GT,
which does not prove the vehicle was actually safe or nonexistent.

## Metric contract

`src/de14/metrics.py` exports
`evaluate_frames(manifest["frames"], predictions, iou_threshold=0.3)`.
Predictions contain one record per supplied frame:

```python
{"clip": "Bago_rural_15", "frame": 58, "vehicles": [
    {"box": [100, 200, 300, 500], "decision": "chưa xác định", "n_riders": 2}
]}
```

The accepted decisions are `không đội mũ`, `có đội mũ`, and `chưa xác định`.
Within each frame, all GT/prediction pairs above the IoU threshold are sorted by
descending IoU, with stable index tie breaking, and matched one-to-one. Duplicate
detections cannot match the same GT twice. Missing prediction frames are empty;
duplicate frame records and predictions outside the manifest raise `ValueError`.

Returned rates use these denominators:

| Rate | Definition |
| --- | --- |
| `vehicle_detection_precision` | Matched vehicles / all predicted vehicles |
| `vehicle_detection_recall` | Matched vehicles / all annotated vehicles |
| `violation_precision` | Matched positive GT predicted violation / all predicted violations, including unmatched predicted violations |
| `violation_recall` | Matched positive GT predicted violation / all positive GT, including missed or unknown predictions |
| `unknown_rate` | Unknown decisions / all predicted vehicles |
| `rider_count_mae` | Absolute rider count error / matched vehicles |
| `rider_count_exact_rate` | Exact rider counts / matched vehicles |

Zero denominators return `None` (JSON `null`), including rider metrics when no
vehicles match. Counts are also returned. These are **frame vehicle metrics**,
not head AP, tracking accuracy, or the track-level metrics in older reports.
Unknown rate alone can hide missed vehicles; interpret it alongside detection
recall and violation recall. Rider metrics measure matched vehicles only.

## Reproduce and verify

From the repository root in PowerShell:

```powershell
.venv/Scripts/python.exe scripts/prepare_benchmark_data.py --ipv4
.venv/Scripts/python.exe tests/test_benchmark_metrics.py
```

The default selection is `--clips 4 --frames-per-clip 4 --seed 20261105`. `--ipv4`
is optional; it was needed here because local IPv6 connection attempts stalled,
while IPv4 reached the same public source successfully. The preparer was rerun
successfully from its cache, preserving the selected frames and counts.

Focused verification: **16 tests passed**, covering overlap matching, global
descending-IoU assignment, duplicate detections, unknown/missing positive GT,
cross-frame isolation, undefined rates, duplicate/extra frame rejection,
deterministic consecutive selection, annotation conversion/duplicate tracks,
fresh-checkout split downloading through fake HTTP, cache reuse, local split
precedence, explicit split selection/missing-file errors, and invalid-download
rejection. Fake HTTP tests use temporary directories and perform no cloud calls.
No model accuracy or inference speed is claimed by this data preparation task.
