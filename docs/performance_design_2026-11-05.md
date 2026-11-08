# Performance and accuracy work — 2026-11-05

## Evidence and scope

Current verified Offline: YOLO26s + full DEIMv2 + head/vehicle association + ByteTrack + DINOv3-S/probe + CLIP. CPU comparison on two known images measured about 3.95/7.24 seconds per image; no independent broad accuracy claim. Current code classifies per vehicle and re-encodes fixed CLIP prompts on every call. ONNX Runtime thread settings are implicit. Local Torch is CPU-only; Intel i7-1355U/Iris Xe is present. Kaggle account credentials and cloud GPU runtime are unavailable in this session.

User authorizes continued development with shared local/cloud code, measured optimization, free/low-cost cloud workflows, and proposals/evidence before architecture changes or retraining.

## Approaches considered

1. Optimize execution of existing models: frame-level crop batching, cached CLIP text embeddings, explicit runtime/thread/device configuration. Selected first: directly supported by code inspection and preserves model semantics; requires output-parity and timing checks.
2. Execute the same full-head ONNX checkpoint with OpenVINO CPU/iGPU. Experimental runtime profile: compare head boxes and end-to-end results before promotion. Do not quantize weights or replace checkpoints in this phase.
3. Replace multiple models with a task-trained detector/classifier. Deferred architecture/training proposal: requires rider/head labels and held-out evaluation; two reported images are insufficient training or acceptance evidence for a new architecture.

## Design

- One shared runtime builder and configuration profiles for baseline CPU, optimized CPU, experimental OpenVINO CPU and explicit CUDA cloud. App and CLI consume the same detector/classifier/pipeline interfaces. Device/provider metadata must reflect actual execution; explicit CUDA must fail if unavailable.
- Keep current association, classifier fusion thresholds, weights and voting. Batch all valid heads per frame, split predictions back to their originating vehicle. Retain baseline per-vehicle execution as a benchmark control.
- CLIP caches normalized text features for its fixed prompt set; compute image features and the same scaled dot products. Preserve uncached forward for a real-model parity control. Support current/older Transformers feature return types.
- ONNX head detector accepts explicit threads. OpenVINO backend uses the same preprocessing/output decoding and checkpoint; request FP32 for numerical comparison and record actual device.
- Benchmark a deterministic official HELMET test subset, excluded from the local training manifest, plus two known regression images. Report frame latency p50/p95, throughput FPS, component timing, actual runtime/providers, weight hashes, classifier/text/image forward counts, CUDA VRAM when available, and frame-level vehicle violation precision/recall with one-to-one GT matching. Undefined metrics are null, never fabricated zeros. Keep source/mirror and annotation limitations.
- Cloud notebook/package executes the same scripts. Do not host a persistent web service on free Colab. No cloud FPS or VRAM claims without a GPU run.
- Additional data: acquire official test frames for evaluation now. Before retraining propose a separate training set covering fabric/fur hats, children, occlusion, distant heads and night/rain, with clip/location splits and manual verification. No model is retrained in this phase.

## Acceptance

- Existing tests remain passing; add tests for batching/order/unknown preservation, runtime validation, cached CLIP parity, metrics one-to-one matching.
- Two user cases keep two violations and rider counts 4+2+1. Compare per-head labels and probabilities, not only totals.
- Optimized profile must improve measured latency and preserve decisions on measured inputs before making it the app default. Keep baseline selectable. OpenVINO stays experimental if regression occurs.
- Changes and benchmark data are reproducible from CLI; precision/recall are identified as frame-level and sampled, not mAP or a general video benchmark. No automatic retraining or benchmark-result transfer from older models.
