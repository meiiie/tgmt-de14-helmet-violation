# Video recognition: preserve measured head observations

## What changed and why

The earlier assisted annotation JSON did not change recognition. This change fixes an actual pipeline loss: `Pipeline.run` previously discarded crops from detections that ByteTrack had not yet confirmed. Confirmation on the next sampled frame could not recover the first real observation. That both shortened the visible track and made its known-vote history harder to reach `n_min`.

The offline pipeline now copies small, measured head crops from tentative detections and retains their source frame/time. If ByteTrack confirms that same identity, the unchanged DINO/probe classifier processes those original crops and inserts the results at the original sampled frame. Unconfirmed/removed candidates are discarded; current images never stand in for past crops. Votes for a past frame use only observations at or before that frame. No interpolation or Kalman-only box becomes a detection or classification.

Final conservative eligibility: only a tentative sample with at least one geometrically valid head crop is buffered. Valid crops classified UNKNOWN are still retained; headless/invalid-only samples are not retrospectively added. This avoids adding unsupported parked-bike boxes for no helmet-recognition benefit. The observation's own head-detector mode is saved so later detector output cannot erase its F2 evidence. Direct streaming/one-pass writers cannot rewrite old frames and retain their earlier behavior; the shared offline two-pass renderer supports the change.

Models, weights, imgsz1280, detector confidence, association geometry, default ByteTrack confidence fusion, n_min and F2 thresholds stay unchanged. CLIP starts unchecked in the UI, honoring the user's earlier preference; DEIM remains enabled. Both OpenVINO CPU and the shared CUDA runtime use this same pipeline. No training or new model was introduced.

## References and alternatives actually tested

[NVIDIA DeepStream tracker](https://docs.nvidia.com/deepstream/text/DS_plugin_gst-nvtracker) documents late activation and optional past-frame data: observations during the tentative period can be retained and reported after activation. This motivates measured-history preservation here, without requiring NVIDIA SDKs on Intel hardware.

[ByteTrack paper](https://arxiv.org/abs/2110.06864) distinguishes high-score detection matching from low-score recovery of existing tracks. It does not justify declaring every weak detection a new object. In this project, changing tracker input geometry, confidence floors and score fusion was first tested on frozen detections:

- Motorcycle-only tracking geometry regressed supplied-box coverage compared with the current vehicle+rider union in several stride settings; not adopted.
- conf_moto .3→.2 produced small/inconsistent changes; not adopted. This experiment cannot recover boxes below the original detector's .2 floor.
- Removing score fusion increased detection coverage, but also substantially changed the number/identity of tracks in 1102. Without valid physical IDs, it remains an explicit constructor-level ablation, not the default.
- Measured-history preservation keeps identity-confirmation rules intact and addresses a demonstrated loss. The head-supported eligibility was selected after the smoke experiment exposed additional headless false positives; this is not a new independent holdout result.

`outputs/video_tracking_20261106/replay_factorial.json` and `replay_backfill.json` preserve exploratory results, including rejected alternatives. Their times are cheap cached replay, never neural inference FPS.

## Official-label smoke evaluation

Data: existing HELMET official-test smoke subset, 4 clips × 4 consecutive frames, 32 labelled vehicle occurrences including 8 violation occurrences. The available mirror is640×640 and resized to1920×1080 to align annotation coordinates; original detail is not recovered. These correlated frames are too small for general accuracy or tracking identity claims.

Both live repeats use the same shared warmed YOLO26s + DEIMv2/OpenVINO CPU + DINOv3-S probe runtime, CLIP off, imgsz1280. Variant order alternates by clip/repeat. Actual delivered labels must use causal vehicle voting, not instantaneous per-head labels.

| Metric, IoU≥.3 | Baseline | Final policy |
|---|---:|---:|
| Violation TP | 6 | 7 |
| Violation FP | 1 | 1 |
| Violation FN | 2 | 1 |
| Violation precision | 85.71% | 87.50% |
| Violation recall | 75.00% | 87.50% |
| Vehicle matches / GT | 26/32 | 27/32 |
| Vehicle precision | 26/55 =47.27% | 27/56 =48.21% |
| Vehicle FP / FN | 29 /6 | 29 /5 |
| UNKNOWN / predictions | 26/55 =47.27% | 26/56 =46.43% |

Both repeats give the same quality counts. Do not confuse better violation detection with accurate total person counting: matched-rider absolute error was13 across26 matches before,14 across27 after. Missing riders/heads remain a separate problem.

Evaluation provenance is deliberately separated:

1. The long-lived live run imported the first, broader backfill implementation. Its raw predictions/reports remain intact under `outputs/video_tracking_20261106/public_*`. Its first-repeat timing overlapped a short unit-test process; exclude that repeat from speed comparisons.
2. Final causal labels are exactly reconstructed from saved instantaneous vehicle VIOL/OK/UNKNOWN votes at n_min1/hi.6/lo.4. UNKNOWN contributes no vote. Histories are sorted by frame and isolated by clip/track/mode/repeat. The later benchmark helper now directly reads `frame_decisions` to prevent this evaluation mismatch.
3. Public F2 wasNone and identity association is unchanged by the final eligibility rule. Only3 newly backfilled zero-head records are removed; existing confirmed records remain. The remaining newly backfilled record has a known classification, proving a valid crop. All IoU matching and metrics are recomputed after removal. This exact downstream projection is labelled **frozen postprocessing evaluation**, not a new live timing run of the final policy.
4. `public_causal_output.json`, per-mode causal prediction JSONs and `score_public_causal.py` retain the complete transformation, source hashes and limitations. Review confirmed the vote reconstruction and the F2=None projection assumptions. The projection must not be applied to filtered user-video output, where changed geometry could change F2; that pipeline is replayed afresh.

Live repeat2 took about70s per16frames on this shared Intel machine (roughly0.23 sampled FPS); detailed per-variant p50/p95 remain in the live report. This does **not** establish a throughput improvement. Most time remains in detector inference. Final-policy replay/render time must not be presented as end-to-end inference speed. VRAM isN/A on this CPU run; no cloud GPU benchmark was performed.

## 1102.mp4 comparison

Use frozen original YOLO/DEIM outputs and original unassisted label snapshot only. New head crops are classified with the same DINO/probe and processor identity; unchanged crops reuse their measured probabilities. Exact baseline parity is asserted for every per-frame record and final track/vote result before evaluating the change. The classifier-only replay avoids loading an unnecessary detector model and does not replace any detector inference measurement.

The final replay and media validation are recorded under `outputs/video_tracking_20261106/final/`. Coverage uses68 supplied head boxes at49 explicitly reviewed anchors, IoU≥.5, after final F2. It is a diagnostic on incomplete labels, not representative recall. No P/R/FP/FN/IDSW is claimed for1102; the assisted draft is never used as GT. Short user IDs are still inconsistent and there are no user-labelled no-helmet positives.

| Stride | Matched supplied heads before →after | Retained vehicle tracks | Restored measured vehicle frames | F2 removed before →after |
|---|---:|---:|---:|---:|
|10|13/68 →17/68|9 →9|8|1 →1|
|5|22/68 →23/68|32 →32|16|9 →9|
|2|29/68 →30/68|62 →63|27|26 →25|

The stride2 change restores29 measured head crops over27 vehicle frames. Added-head classification took3.80s in the replay (already measured detector outputs were reused); this is extra necessary work, not an inference speedup. The UNKNOWN track counts at n_min5 remain7/27/48 for strides10/5/2; stride2 denominator grows62→63. Predicted violation-track counts remain0/0/3. Do not interpret those3 as verified violations without reviewed temporal GT. F2 now keeps one additional stride2 track because actual old head evidence is preserved; its audit remains available for review.

The full-length output keeps1052frames,30FPS,1920×1080 and source audio; unsampled frames stay original/unannotated. Observations restored to sampled frames are measured source crops, not interpolation. Media/frame count and decoded-audio checks are recorded separately. The default UI/CLI automatically uses the new head-supported offline history on the next launch.

Validated output: `outputs/video_tracking_20261106/final/1102_backfill_stride2_full_length.mp4`,35.0666667s, SHA256 `c83c0da42ebd246f70436f7de5a447027a7a609d3caf9560dd7b38be7076132f`. Source and output decoded audio both hash to `dff75fb03263d33ebc5a853853214d3da521e56c44da04559670b7ae6caa79f0`. `final_acceptance.json` and `docs/results/video_tracking_2026-11-06.json` contain compact evidence; source hashes, original baseline parity and final model/processor identity are explicit. Python in-memory tuple/list processor fields are compared through canonical JSON to match their persisted representation; loaded tensor hashes match exactly.

## Verification and remaining limits

`tests/test_tentative_observations.py` uses actual ByteTrack with controlled detector/classifier boundaries: correct old crop and source time, reused decoder arrays, causal voting, one vote per multi-head vehicle frame, UNKNOWN retention, abandoned candidates, invalid/headless eligibility, streaming suppression, F2 evidence and actual-output scoring. Failing-before/passing-after transcripts are retained. The full suite run is recorded in `outputs/video_tracking_20261106/tests_final.log`; unit counts are software regression evidence, not accuracy measurements on that many real videos.

The fix cannot create a motorcycle the detector missed or recover a head already rejected by association. For example, source20's low-confidence motorcycle0.0721 lies below the current YOLO threshold; source10 lacked a motorcycle even in the earlier .01 diagnostic. Those cases require a separate controlled detector/recovery experiment or labelled fine-tuning; this change does not claim to solve them. Keep reviewing crowded-rider data and physical IDs before changing association/thresholds or claiming identity continuity. Never lower min observations merely to make UNKNOWN disappear.
