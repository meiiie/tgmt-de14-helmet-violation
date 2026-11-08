"""Đề 14 – Gradio demo: phát hiện người đi xe máy không đội mũ bảo hiểm (ảnh / video).
CPU mặc định: YOLO26s (xe + người) + ByteTrack + phân loại đầu (YOLO helmet hoặc DINOv3-S probe).
GPU tuỳ chọn: SAM 3.1 (cần checkpoint gated qua HF_TOKEN hoặc SAM31_CKPT).
Chạy: python app.py   (Colab/Kaggle: python app.py --share)"""
import argparse, os, sys, csv, tempfile, time
import cv2, numpy as np, gradio as gr
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from de14.core import VIOL, OK, UNDET

LBL = {VIOL: "không mũ", OK: "có mũ", UNDET: "chưa xác định"}
BACKENDS = ["YOLO26s + YOLO helmet", "YOLO26s + DINOv3-S probe", "SAM 3.1 + DINOv3-S probe (GPU)"]
PUBLIC_SHARE = False
import json as _json, torch as _torch
# Cấu hình mặc định chọn ở kernel v4 (F1 cao nhất với P tính cả track không khớp): SAM 3.1 4 prompt + DINOv3-S + bộ lọc F2
TRACK_FILTER = _json.load(open("configs/track_filter.json"))["track_filter"]
DINO_PROBES = ("weights/dino_probe.joblib", "weights/dino_probe_dinov3-vits16-pretrain-lvd1689m.joblib")
ACCURACY_HEAD = "weights/deimv2_wholebody49_boxes_only.onnx"
ACCURACY_CLIP = "weights/clip-vit-base-patch32"


def _openvino_devices():
    try:
        import openvino as ov
        return tuple(ov.Core().available_devices)
    except Exception:
        return ()


OPENVINO_DEVICES = _openvino_devices()


def _available_backends():
    return BACKENDS if _torch.cuda.is_available() else BACKENDS[:2]


def _available_runtime_choices():
    choices = [("CPU đối chứng", "baseline_cpu"), ("CPU tối ưu", "optimized_cpu")]
    devices = {d.split('.')[0] for d in OPENVINO_DEVICES}
    if "CPU" in devices:
        choices.append(("Intel OpenVINO CPU", "openvino_cpu"))
    if "GPU" in devices:
        choices.append(("Intel Iris Xe / OpenVINO GPU (thử nghiệm)", "openvino_gpu"))
    if _torch.cuda.is_available():
        choices.append(("NVIDIA CUDA", "cloud_cuda"))
    return choices


def _default_runtime_profile():
    available = {value for _, value in _available_runtime_choices()}
    configured = os.environ.get("HELMET_RUNTIME_PROFILE")
    if configured in available:
        return configured
    if "cloud_cuda" in available:
        return "cloud_cuda"
    return "openvino_cpu" if "openvino_cpu" in available else "optimized_cpu"


def _head_ready():
    return os.path.isfile(ACCURACY_HEAD)

def _accuracy_ready():
    return _head_ready() and os.path.isfile(os.path.join(ACCURACY_CLIP, "model.safetensors"))

def _default_backend():
    if not any(os.path.isfile(p) for p in DINO_PROBES):
        return BACKENDS[0]
    return BACKENDS[1]

def _require_dino_probe():
    for path in DINO_PROBES:
        if os.path.isfile(path): return path
    raise gr.Error(
        "Thiếu weights/dino_probe.joblib (probe DINOv3 đã huấn luyện). "
        "Chép probe đã huấn luyện vào weights/ hoặc tạo bằng python src/build_head_crops.py "
        "rồi python src/train_probe.py sau khi chuẩn bị dữ liệu HELMET và quyền truy cập DINOv3. "
        "Để demo ngay, chọn tab Offline và backend YOLO26s + YOLO helmet (CPU)."
    )

DEFAULT_BACKEND = _default_backend()
_cache = {}

def get_runtime(backend, imgsz, accurate=False, runtime_profile=None):
    from de14.runtime import build_runtime, load_profile
    profile = runtime_profile or _default_runtime_profile()
    try:
        config = load_profile(profile)
    except ValueError as error:
        raise gr.Error(str(error)) from error
    if (backend.startswith("SAM") or config.device.startswith("cuda")) and not _torch.cuda.is_available():
        raise gr.Error("Runtime hiện tại không có NVIDIA CUDA khả dụng. Trên máy Intel, chọn YOLO26s + DINOv3-S probe "
                       "và Intel OpenVINO CPU (hoặc CPU tối ưu). Iris Xe dùng OpenVINO, không dùng CUDA. "
                       "Để chạy SAM 3.1, hãy chạy app trên môi trường NVIDIA GPU có Torch CUDA và trọng số SAM; "
                       "lựa chọn này không tự kết nối tới GPU cloud.")
    if not backend.startswith("SAM") and config.head_backend == "openvino":
        family = config.openvino_device.split('.')[0]
        if not any(d.split('.')[0] == family for d in OPENVINO_DEVICES):
            raise gr.Error(f"OpenVINO chưa phát hiện thiết bị {config.openvino_device}. Hãy chọn CPU tối ưu hoặc thiết bị đang có trong danh sách.")
    key = (backend, imgsz, bool(accurate), profile)
    if key in _cache: return _cache[key]
    probe_path = _require_dino_probe() if "DINOv3" in backend else DINO_PROBES[0]
    if accurate and not _accuracy_ready():
        raise gr.Error("Run python scripts/prepare_accuracy_models.py to prepare the optional accuracy weights.")
    overrides = {"imgsz": int(imgsz), "accurate": bool(accurate), "detect_heads": _head_ready(), "probe_path": probe_path,
                 "classifier_kind": "yolo" if "YOLO helmet" in backend else "dino",
                 "detector_kind": "sam3.1" if backend.startswith("SAM") else "yolo"}
    if backend.startswith("SAM"): overrides.update(device="cuda", head_backend="onnx")
    try:
        bundle = build_runtime(load_profile(profile, **overrides), reuse=_cache.get((backend, imgsz, not bool(accurate), profile)))
    except (RuntimeError, ValueError) as error:
        raise gr.Error(str(error)) from error
    if len(_cache) >= 2: _cache.pop(next(iter(_cache)))
    _cache[key] = bundle
    return bundle


def get_models(backend, imgsz, accurate=False, runtime_profile=None):
    bundle = get_runtime(backend, imgsz, accurate, runtime_profile)
    return bundle.detector, bundle.classifier

def _remote_options(video, backend, stride, n_min, imgsz, use_filter, accurate, max_sec, output_mode):
    from de14.remote_jobs import JobOptions
    if backend not in BACKENDS:raise gr.Error("Backend không hợp lệ.")
    return JobOptions(kind="video" if video else "image",detector="sam3.1" if backend.startswith("SAM") else "yolo",
                       classifier="yolo" if "YOLO helmet" in backend else "dino",imgsz=int(imgsz),
                       accurate=bool(accurate),detect_heads=True,stride=int(stride),n_min=int(n_min),
                       use_filter=bool(use_filter),max_seconds=float(max_sec or 0),output_mode=output_mode)


def export_colab_job(image, video, backend, stride, n_min, imgsz, use_filter=True, accurate=False, max_sec=0, output_mode="full"):
    from pathlib import Path
    from de14.remote_jobs import create_request
    if bool(image) == bool(video):
        raise gr.Error("Chọn đúng một ảnh hoặc một video để gửi sang Colab.")
    options = _remote_options(video, backend, stride, n_min, imgsz, use_filter, accurate, max_sec, output_mode)
    folder=Path(tempfile.mkdtemp(prefix="de14_colab_"));archive=folder/"helmet_gpu_request.zip"
    manifest=create_request(video or image,options,archive)
    state={"job_id":manifest['job_id'],"input_sha256":manifest['input']['sha256']}
    return str(archive),state,"Đã tạo job. Tải ZIP và code mới, chạy ô nhận diện trong Colab T4 rồi nhập ZIP kết quả ở bước 3. Chưa chạy inference trên máy local."


def import_colab_result(archive, state=None):
    from pathlib import Path
    from zipfile import ZipFile,BadZipFile
    from de14.remote_jobs import read_result
    if not archive:raise gr.Error("Chọn ZIP kết quả ảnh/video được Colab tạo sau khi nhận diện.")
    try:
        with ZipFile(archive) as z:
            if 'result.json' not in z.namelist():
                raise gr.Error("Đây không phải ZIP kết quả ảnh/video. ZIP benchmark chỉ chứa số đo; hãy chạy ô nhận diện GPU trong Colab để tạo helmet_gpu_result_....zip.")
        folder=Path(tempfile.mkdtemp(prefix="de14_colab_result_"))
        manifest,root=read_result(archive,folder,(state or {}).get('job_id'),(state or {}).get('input_sha256'))
        result=manifest['result'];meta=result['runtime'];relative=result['media_path']
        names={f['path'] for f in manifest['files']}
        if relative not in names:raise ValueError("Missing annotated media in result")
        media=root.parent/relative;kind=result['kind'];csv_name=result.get('tracks_path')
        if kind not in {'image','video'}:raise ValueError("Invalid result media kind")
        if not str(meta.get('torch_device','')).startswith('cuda'):raise ValueError("Kết quả này không báo chạy trên CUDA GPU")
        counts=result['counts'];timing=result['timings']
        text=(f"**{counts['vehicles']} xe/track** · **{counts['people']} người phát hiện** · xe có người không mũ: **{counts['violations']}** · chưa xác định: {counts.get('unknown',0)} · "
              f"GPU: {meta.get('cuda_device','NVIDIA CUDA')} · "
              f"{timing['sampled_frames']} frame phân tích · {timing['processing_seconds']:.1f}s xử lý theo báo cáo GPU.")
        if timing.get('model_load_seconds') is not None:
            text += f" · tải model: {timing['model_load_seconds']:.1f}s"
        untracked = result.get('untracked_head_retention', {}).get('heads', 0)
        if untracked:
            text += f" · {untracked} lượt đầu ở xe chưa xác nhận track (UNCONFIRMED)"
        return (str(media) if kind=='image' else None,str(media) if kind=='video' else None,text,
                str(root.parent/csv_name) if csv_name and csv_name in names else None)
    except gr.Error:raise
    except (ValueError,KeyError,TypeError,BadZipFile,OSError) as error:
        raise gr.Error("Không thể nạp kết quả Colab: "+str(error)) from error


def analyze(image, video, backend, stride, max_sec, n_min, imgsz, use_filter=True, accurate=False, runtime_profile=None, output_mode="full"):
    if image is None and video is None:
        raise gr.Error("Hãy tải lên ảnh hoặc video")
    load_start = time.perf_counter()
    runtime = get_runtime(backend, int(imgsz), accurate, runtime_profile)
    model_load_seconds = time.perf_counter()-load_start
    out = tempfile.mkdtemp(prefix="de14_")
    t0 = time.time()
    if video is not None:
        from de14.video_io import run_video
        pl = runtime.pipeline(n_min=int(n_min), track_filter=TRACK_FILTER if use_filter else None); name = "video"
        r = run_video(video, pl, out, name, stride=int(stride), max_seconds=float(max_sec or 0), output_mode=output_mode)
        with open(os.path.join(out, "video_runtime.json"), "w", encoding="utf-8") as file:
            _json.dump({"runtime": runtime.metadata(), "model_load_seconds": model_load_seconds}, file, ensure_ascii=False, indent=2)
    else:
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        pl = runtime.pipeline(fps=1, n_min=1); name = "image"
        r = pl.run(iter([(0, img)]), out_dir=out, video_name=name, tracking=False, write_video=False, render=True)
    el = time.time() - t0
    rows, gallery = [], []
    csv_path = os.path.join(out, f"{name}_tracks.csv")
    with open(csv_path, encoding="utf-8") as file:
        for row in csv.DictReader(file):
            conf = "" if row["no_helmet_ratio"] in ("", "None") else f"{float(row['no_helmet_ratio']):.2f}"
            rows.append([int(row["track_id"]), f"{float(row['t_start']):.1f}–{float(row['t_end']):.1f}s", LBL.get(row["decision"], row["decision"]), conf, int(row["n_obs"]), int(row["n_riders"])])
            if row["evidence"]:
                gallery.append((os.path.join(out, row["evidence"]), f"#{row['track_id']} {LBL.get(row['decision'])} t={float(row['t_start']):.1f}s"))
    rows.sort(key=lambda x: (x[2] != "không mũ", x[0]))
    vis_img, vis_vid = None, None
    if video is not None: vis_vid = r["video_path"]
    elif pl.last_vis is not None: vis_img = cv2.cvtColor(pl.last_vis, cv2.COLOR_BGR2RGB)
    n_v = sum(1 for x in rows if x[2] == "không mũ"); n_u = sum(1 for x in rows if x[2] == "chưa xác định")
    n_people = sum(row[5] for row in rows)
    summary = (f"**{len(rows)} xe/track** · **{n_people} người phát hiện** · xe có người không mũ: **{n_v}** · chưa xác định: {n_u} · {r['frames']} frame · "
               f"{el:.1f}s xử lý ({r['frames']/max(el,1e-6):.2f} frame phân tích/s) · backend: {backend}")
    summary += f" · runtime: {runtime_profile or _default_runtime_profile()} · device: {runtime.device}"
    summary += " · đầu: " + ("DEIM" if runtime.config.head_enabled else "ước lượng từ người")
    if video is not None and r.get("temporal_backfill", {}).get("vehicle_frames"):
        summary += f" · khôi phục {r['temporal_backfill']['vehicle_frames']} lượt phát hiện ở đầu track từ frame đã phân tích"
    untracked = r.get("untracked_head_retention", {}).get("heads", 0)
    if untracked:
        summary += f" · {untracked} lượt đầu ở xe chưa xác nhận track (UNCONFIRMED)"
    classifier_name = "DINO" if runtime.config.classifier_kind == "dino" else "YOLO helmet"
    clip_enabled = runtime.config.accurate and runtime.config.verify_with_clip
    summary += " · phân loại: " + classifier_name + (" + CLIP đối chiếu" if clip_enabled else ", không đối chiếu CLIP")
    if video is not None:
        media = r["media"]; output = media["output"]
        mode_label = "Full-length Annotated Video" if output_mode == "full" else "Sampled Preview"
        fps_label = "FPS trung bình (VFR)" if output.get("vfr") else "FPS"
        summary += (f" · {mode_label}: {output['video_duration']:.2f}s / {output['fps']:.3f} {fps_label} / "
                    f"{output['width']}×{output['height']} / {media['written_frames']} frame"
                    f" · {r['frames']} frame inference ({r['fps']:.2f} sampled FPS)"
                    f" · F2: {r['f2']['total_before']} trước lọc, {r['f2']['removed']} loại, {r['f2']['kept']} giữ"
                    f" · tải model: {model_load_seconds:.1f}s · frame không phân tích giữ nguyên, không annotation")
        if media["stop_reason"] != "source_end":
            summary += f" · chỉ phân tích đến {media['analyzed_last_timestamp_s']:.2f}s; video full vẫn giữ toàn bộ nguồn"
    return vis_img, vis_vid, summary, rows, gallery, csv_path

# ---------------- Realtime ----------------
RT_DEFAULT = {"imgsz": 640, "every_n": 5}   # cập nhật theo kết quả đo kernel v6 (README mục G)
def _rt_pipeline(w, sz, n, f2):
    probe_path = _require_dino_probe()
    from de14.realtime import RealtimePipeline
    return RealtimePipeline(f"weights/{w}.pt", probe_path, imgsz=int(sz), every_n=int(n),
                            track_filter=TRACK_FILTER if f2 else None, fps=25)

def rt_stream(src, file, w, sz, n, f2):
    source = (src or "").strip()
    if PUBLIC_SHARE and source:
        raise gr.Error("Liên kết chia sẻ chỉ nhận video tải lên.")
    path = source or file
    if not path: raise gr.Error("Nhập URL RTSP/HTTP, đường dẫn hoặc tải file lên")
    cap = cv2.VideoCapture(path); pl = _rt_pipeline(w, sz, n, f2); t0 = time.time(); k = 0; ema = None; tl = time.time()
    try:
        while True:
            ok, f = cap.read()
            if not ok: break
            recs = pl.step(f); k += 1; now = time.time(); inst = 1 / max(1e-6, now - tl); tl = now
            ema = inst if ema is None else 0.9 * ema + 0.1 * inst
            vis = pl.draw(f.copy(), recs, fps=ema)
            nv = sum(1 for t in pl.stats if pl.decision(t)[0] == VIOL)
            yield cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), f"frame {k} · FPS TB {k/(now-t0):.1f} · track {len(pl.stats)} · không mũ {nv} · hàng đợi DINO {pl.q.qsize()}"
    finally:
        cap.release(); pl.close()

def rt_webcam(frame, st, w, sz, n, f2):
    if frame is None: return None, st, ""
    key = (w, int(sz), int(n), bool(f2))
    if st is None or st["key"] != key:
        if st: st["pl"].close()
        st = {"key": key, "pl": _rt_pipeline(w, sz, n, f2), "t": time.time(), "ema": None}
    f = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR); recs = st["pl"].step(f); now = time.time()
    inst = 1 / max(1e-6, now - st["t"]); st["t"] = now; st["ema"] = inst if st["ema"] is None else 0.9 * st["ema"] + 0.1 * inst
    vis = st["pl"].draw(f.copy(), recs, fps=st["ema"])
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), st, f"FPS {st['ema']:.1f} · track {len(st['pl'].stats)}"


def _short_result_summary(text):
    """Keep the viewer's counts up front; full processing report remains available."""
    return ' · '.join((text or '').split(' · ')[:4]) or 'Kết quả sẽ xuất hiện sau khi nhận diện.'


def _show_result_media(image,video):
    return gr.update(visible=image is not None),gr.update(visible=video is not None)


def _gpu_client(url='', token=''):
    from de14.gpu_client import GpuApiClient
    configured = os.environ.get('HELMET_GPU_API_URL','').strip().rstrip('/')
    endpoint = (url or configured).strip().rstrip('/')
    if not endpoint:
        raise ValueError('Chưa cấu hình GPU API. Mở Cài đặt nâng cao để nhập địa chỉ backend NVIDIA.')
    if PUBLIC_SHARE and endpoint != configured:
        raise ValueError('Liên kết chia sẻ chỉ dùng GPU API do máy chủ cấu hình.')
    # Never forward the server-side secret to a user-overridden endpoint.
    key = token or (os.environ.get('HELMET_GPU_API_TOKEN','') if endpoint == configured else '')
    return GpuApiClient(endpoint,key)


def check_gpu_connection(url, token):
    try:
        info = _gpu_client(url,token).health()
        return f"**GPU đã kết nối: {info['gpu']}** · {info['pending_jobs']}/{info['max_pending']} job trong hàng đợi."
    except (ValueError,RuntimeError,OSError) as error:
        return '**GPU chưa kết nối.** ' + str(error)


def analyze_web(image, video, backend, stride, max_sec, n_min, imgsz, use_filter=True,
                accurate=False, runtime_profile=None, output_mode='full', execution='gpu_api',
                api_url='', api_token='', progress=gr.Progress()):
    if execution == 'local':
        return analyze(image,video,backend,stride,max_sec,n_min,imgsz,use_filter,accurate,runtime_profile,output_mode)
    if execution not in {'gpu_api', 'kaggle'}:raise gr.Error('Chế độ xử lý không hợp lệ.')
    if execution == 'kaggle' and not kaggle_configured:
        raise gr.Error('Kaggle chưa được cấu hình trên máy chạy web.')
    if (image is None) == (video is None):raise gr.Error('Chọn đúng một ảnh hoặc một video.')
    from pathlib import Path
    from de14.remote_jobs import create_request
    service = 'Kaggle' if execution == 'kaggle' else 'GPU API'
    try:
        started=time.perf_counter()
        if execution == 'kaggle' and (backend != BACKENDS[1] or accurate):
            raise gr.Error('Kaggle dùng YOLO26s + DINOv3-S probe với CLIP tắt.')
        options = _remote_options(video, backend, stride, n_min, imgsz, use_filter, accurate, max_sec, output_mode)
        progress(0, desc=f'Chuẩn bị job {service}')
        with tempfile.TemporaryDirectory(prefix='de14_web_job_') as folder:
            work = Path(folder)
            media = Path(video) if video is not None else work / 'image.png'
            if image is not None:
                image.save(media)
            archive = work / 'request.zip'
            manifest = create_request(media, options, archive)
            state = {'job_id': manifest['job_id'], 'input_sha256': manifest['input']['sha256']}
            if execution == 'kaggle':
                from de14.kaggle_jobs import KaggleBatch
                result, kaggle_timing = KaggleBatch().run(archive, state['job_id'], work, progress)
            else:
                client = _gpu_client(api_url, api_token)
                client.health()
                progress(.1, desc='Gửi file đến GPU')
                client.submit(archive, state['job_id'])
                progress(.2, desc='GPU đang xử lý')
                client.wait(state['job_id'])
                progress(.9, desc='Nhận và kiểm tra kết quả')
                result = client.download(state['job_id'], work / 'result.zip')
            img,vid,report,csv_path=import_colab_result(result,state)
        if execution == 'kaggle':
            report += f" · tải lên Kaggle {kaggle_timing['upload_seconds']:.1f}s"
            report += f" · chuẩn bị dataset {kaggle_timing['dataset_ready_seconds']:.1f}s"
            wait = kaggle_timing['queue_wait_observed_seconds']
            if wait is not None:
                report += f" · chờ GPU khoảng {wait:.1f}s"
            environment = kaggle_timing['environment']
            if environment:
                preparation = sum(environment[key] for key in ('inputs_seconds', 'cuda_check_seconds', 'dependencies_seconds'))
                report += f" · chuẩn bị trên Kaggle {preparation:.1f}s (thư viện {environment['dependencies_seconds']:.1f}s)"
            report += f" · tải kết quả {kaggle_timing['download_seconds']:.1f}s"
        rows=[];gallery=[]
        if csv_path:
            root=Path(csv_path).parent
            with open(csv_path,encoding='utf8') as file:
                for row in csv.DictReader(file):
                    conf='' if row['no_helmet_ratio'] in ('','None') else f"{float(row['no_helmet_ratio']):.2f}"
                    rows.append([int(row['track_id']),f"{float(row['t_start']):.1f}–{float(row['t_end']):.1f}s",
                                 LBL.get(row['decision'],row['decision']),conf,int(row['n_obs']),int(row['n_riders'])])
                    evidence=(root/row.get('evidence','')).resolve()
                    if row.get('evidence') and evidence.is_relative_to(root.resolve()) and evidence.is_file():
                        gallery.append((str(evidence),f"#{row['track_id']} {LBL.get(row['decision'])}"))
            rows.sort(key=lambda x:(x[2]!='không mũ',x[0]))
        report+=f" · tổng thời gian qua {service}: {time.perf_counter()-started:.1f}s (gồm truyền file và chờ GPU)"
        progress(1,desc='Hoàn tất')
        return img,vid,report,rows,gallery,csv_path
    except gr.Error:raise
    except (ValueError,RuntimeError,OSError,TimeoutError) as error:
        raise gr.Error(f'Không thể xử lý qua {service}: {error}') from error


configured_gpu_url = os.environ.get('HELMET_GPU_API_URL', '').strip()
kaggle_configured = bool(os.environ.get('HELMET_KAGGLE_NOTEBOOK') and os.environ.get('HELMET_KAGGLE_MODELS'))
default_execution = 'gpu_api' if configured_gpu_url and not kaggle_configured else 'local'
execution_choices = [('GPU qua API', 'gpu_api'), ('Máy này (CPU / OpenVINO / CUDA có sẵn)', 'local')]
if kaggle_configured:
    execution_choices.insert(0, ('GPU Kaggle', 'kaggle'))

EVIDENCE_GALLERY_CSS = """
#evidence-gallery .grid-wrap {
    min-height: 0;
    max-height: none;
    overflow: visible;
    padding-top: 2rem;
}
#evidence-gallery .grid-container {
    grid-template-columns: repeat(auto-fill, minmax(min(100%, 8rem), 1fr));
    grid-auto-rows: auto;
}
#evidence-gallery .thumbnail-lg {
    display: flex;
    flex-direction: column;
    aspect-ratio: auto;
    height: auto;
}
#evidence-gallery .thumbnail-lg > img {
    height: auto;
    aspect-ratio: 4 / 3;
}
#evidence-gallery .thumbnail-lg .caption-label {
    position: static;
    max-width: none;
    white-space: normal;
    border-left: 0;
    opacity: 1;
}
"""

with gr.Blocks(title="Kiểm tra mũ bảo hiểm") as demo:
    gr.Markdown("# Kiểm tra mũ bảo hiểm\n"
                "Chọn ảnh hoặc video để nhận diện. **Xanh:** có mũ · **Đỏ:** không mũ · **Vàng:** cần kiểm tra thêm.")
    with gr.Tab("Nhận diện"):
        connection = gr.Markdown({
            'kaggle': '**GPU Kaggle** · Có thể phải chờ Kaggle cấp GPU.',
            'gpu_api': '**GPU API đã được cấu hình.**',
            'local': '**Xử lý trên máy này.**',
        }[default_execution])
        with gr.Row():
            with gr.Column(scale=1):
                img_in = gr.Image(label="Chọn ảnh", type="pil",height=220)
                vid_in = gr.Video(label="Hoặc chọn video",height=220)
                btn = gr.Button("Nhận diện", variant="primary")
                with gr.Accordion("Cài đặt nâng cao",open=False):
                    execution=gr.Radio(execution_choices,value=default_execution,label='Nơi xử lý')
                    with gr.Accordion('Kết nối máy chủ GPU',open=False):
                        api_url=gr.Textbox(value=configured_gpu_url,label='Địa chỉ GPU API',placeholder='https://gpu.example.com')
                        api_token=gr.Textbox(value='',type='password',label='Khóa API (để trống nếu máy chủ web đã cấu hình)')
                        check_connection=gr.Button('Kiểm tra kết nối GPU')
                    backend = gr.Dropdown(BACKENDS, value=DEFAULT_BACKEND if default_execution == 'local' else BACKENDS[1],
                                          label="Bộ nhận diện")
                    runtime_profile = gr.Dropdown(_available_runtime_choices(), value=_default_runtime_profile(), label="Thiết bị khi chọn Máy này")
                    with gr.Row():
                        stride = gr.Slider(1, 10, value=3, step=1, label="Lấy mẫu mỗi N frame (video)")
                        max_sec = gr.Number(value=0, minimum=0, label="Giới hạn phân tích (giây, 0 = toàn bộ video)")
                    output_mode = gr.Dropdown([("Full-length Annotated Video", "full"), ("Sampled Preview", "sampled_preview")],value="full",label="Video đầu ra")
                    with gr.Row():
                        n_min = gr.Slider(1, 10, value=3, step=1, label="Quan sát hợp lệ tối thiểu / track")
                        imgsz = gr.Dropdown([640, 960, 1280], value=1280, label="imgsz YOLO")
                    use_filter = gr.Checkbox(value=True, label="F2: lọc track ít dịch chuyển và ít bằng chứng người/đầu")
                    accurate = gr.Checkbox(value=False, label="Đối chiếu mũ vải bằng CLIP (thận trọng hơn, chậm hơn)")
                    gr.Markdown("Mặc định dùng bộ tìm đầu DEIM khi có trọng số, giữ video đầy đủ và âm thanh. Frame không phân tích giữ nguyên. "
                                "CLIP đang tắt. Nhãn UNCONFIRMED đánh dấu kết quả từng frame ở xe chưa xác nhận track. "
                                "Báo cáo vi phạm chỉ tổng hợp các track đã xác nhận.")
            with gr.Column(scale=2):
                summary_brief = gr.Markdown("Kết quả sẽ xuất hiện sau khi nhận diện.")
                out_vid = gr.Video(label="Kết quả video",height=420,visible=False)
                out_img = gr.Image(label="Kết quả ảnh",height=420,visible=False)
                csv_out = gr.File(label="Tải báo cáo CSV")
                with gr.Accordion("Thông tin xử lý",open=False):
                    summary = gr.Markdown()
        with gr.Accordion("Bảng chi tiết và ảnh bằng chứng",open=False):
            table = gr.Dataframe(headers=["track_id", "thời gian", "nhãn", "tỉ lệ không mũ", "số quan sát", "số người phát hiện (tối đa)"], label="Bảng vi phạm / theo track")
            gallery = gr.Gallery(
                label="Ảnh bằng chứng (crop đầu)", elem_id="evidence-gallery",
                height="auto", object_fit="contain", fit_columns=False,
            )
        check_connection.click(check_gpu_connection,[api_url,api_token],[connection],api_name=False)
        btn.click(analyze_web, [img_in, vid_in, backend, stride, max_sec, n_min, imgsz, use_filter, accurate, runtime_profile, output_mode, execution, api_url, api_token], [out_img, out_vid, summary, table, gallery, csv_out], api_name="analyze_web").success(
            _short_result_summary,inputs=[summary],outputs=[summary_brief],api_name=False).success(
            _show_result_media,inputs=[out_img,out_vid],outputs=[out_img,out_vid],api_name=False)

    with gr.Tab("GPU Colab (ảnh / video)",visible=False):
        gr.Markdown("## Chuyển ảnh/video sang Colab\nTiện ích này vẫn cần chạy ô notebook và chuyển ZIP thủ công; chưa phải kết nối GPU trực tiếp.")
        with gr.Accordion("Hướng dẫn và file chuẩn bị Colab",open=False):
            gr.Markdown("Tạo job → chạy ô mã trong notebook T4 đang mở → nhập ZIP kết quả bên phải. Dùng lại môi trường/model đã cài, không chạy lại benchmark.")
            with gr.Row():
                gr.File(value="outputs/cloud/helmet_workspace.zip" if os.path.isfile("outputs/cloud/helmet_workspace.zip") else None,
                        label="Code mới để cập nhật Colab (giữ model đã tải)")
                gr.File(value="outputs/cloud/colab_gpu_inference_cell.py" if os.path.isfile("outputs/cloud/colab_gpu_inference_cell.py") else None,
                        label="Ô mã để dán vào notebook Colab hiện tại")
        with gr.Row():
            with gr.Column(scale=1):
                cloud_image=gr.Image(type="filepath",label="Ảnh gửi sang Colab")
                cloud_video=gr.Video(label="Hoặc video gửi sang Colab")
                cloud_export=gr.Button("1 · Tạo job gửi Colab",variant="primary")
                cloud_request=gr.File(label="Tải job ZIP này lên Colab")
                cloud_state=gr.State(None);cloud_status=gr.Markdown()
                with gr.Accordion("Cài đặt GPU nâng cao",open=False):
                    cloud_backend=gr.Dropdown(BACKENDS,value=BACKENDS[1],label="Backend chạy trên GPU Colab")
                    gr.Markdown("SAM 3.1 cần có thêm trọng số/quyền truy cập SAM trong Colab.")
                    with gr.Row():
                        cloud_stride=gr.Slider(1,10,value=3,step=1,label="Lấy mẫu mỗi N frame")
                        cloud_min=gr.Slider(1,10,value=3,step=1,label="Quan sát hợp lệ tối thiểu / track")
                        cloud_size=gr.Dropdown([640,960,1280],value=1280,label="imgsz GPU")
                    cloud_filter=gr.Checkbox(value=True,label="F2 cho video GPU")
                    cloud_clip=gr.Checkbox(value=False,label="Đối chiếu CLIP trên GPU")
                    cloud_limit=gr.Number(value=0,minimum=0,label="Giới hạn phân tích GPU (giây, 0 = toàn bộ)")
                    cloud_mode=gr.Dropdown([("Video đầy đủ, giữ FPS và âm thanh","full"),("Sampled Preview","sampled_preview")],value="full",label="Video GPU đầu ra")
            with gr.Column(scale=2):
                cloud_result=gr.File(file_types=[".zip"],type="filepath",label="3 · Chọn helmet_gpu_result_....zip Colab trả về")
                cloud_import=gr.Button("Mở kết quả GPU")
                cloud_brief=gr.Markdown()
                cloud_out_video=gr.Video(label="Kết quả video GPU",height=420)
                cloud_out_image=gr.Image(label="Kết quả ảnh GPU",height=280)
                cloud_csv=gr.File(label="CSV tracks từ GPU")
                with gr.Accordion("Thông tin xử lý GPU",open=False):
                    cloud_summary=gr.Markdown()
        cloud_export.click(export_colab_job,[cloud_image,cloud_video,cloud_backend,cloud_stride,cloud_min,cloud_size,cloud_filter,cloud_clip,cloud_limit,cloud_mode],
                           [cloud_request,cloud_state,cloud_status],api_name="export_colab_job")
        cloud_import.click(import_colab_result,[cloud_result,cloud_state],[cloud_out_image,cloud_out_video,cloud_summary,cloud_csv],api_name="import_colab_result").success(
            _short_result_summary,inputs=[cloud_summary],outputs=[cloud_brief],api_name=False)

    with gr.Tab("Camera / RTSP (thử nghiệm)",visible=False):
        gr.Markdown("YOLO26s chạy **mọi frame** + ByteTrack; DINOv3-S phân loại đầu **bất đồng bộ** (track mới hoặc mỗi N frame); "
                    "F2 tính online (xám = bị lọc F2). Góc dưới trái hiển thị FPS end-to-end và độ dài hàng đợi.")
        with gr.Row():
            with gr.Column():
                rt_src = gr.Textbox(label="RTSP/HTTP URL hoặc đường dẫn file (để trống nếu dùng file tải lên / webcam)")
                rt_file = gr.Video(label="File video (phát dạng stream)")
                rt_cam = gr.Image(sources=["webcam"], streaming=True, label="Webcam", type="numpy")
                with gr.Row():
                    rt_w = gr.Dropdown(["yolo26s", "yolo11s"], value="yolo26s", label="Detector")
                    rt_sz = gr.Dropdown([640, 960, 1280], value=RT_DEFAULT["imgsz"], label="imgsz")
                    rt_n = gr.Slider(1, 15, value=RT_DEFAULT["every_n"], step=1, label="Phân loại lại mỗi N frame / track")
                rt_f2 = gr.Checkbox(value=True, label="F2 online")
                rt_go = gr.Button("Bắt đầu stream (URL / file)", variant="primary")
            with gr.Column():
                rt_out = gr.Image(label="Kết quả realtime", streaming=True)
                rt_info = gr.Markdown()
        rt_state = gr.State(None)
        rt_go.click(rt_stream, [rt_src, rt_file, rt_w, rt_sz, rt_n, rt_f2], [rt_out, rt_info])
        rt_cam.stream(rt_webcam, [rt_cam, rt_state, rt_w, rt_sz, rt_n, rt_f2], [rt_out, rt_state, rt_info], stream_every=0.1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--share", action="store_true"); ap.add_argument("--port", type=int, default=7861, help="Server port (default: 7861)")
    a = ap.parse_args()
    if kaggle_configured and a.share:
        ap.error("Chế độ Kaggle chỉ chạy trên localhost vì dùng tài khoản Kaggle của máy chủ.")
    PUBLIC_SHARE = a.share
    demo.queue().launch(server_name="127.0.0.1",
                        server_port=a.port, share=a.share, show_error=True, css=EVIDENCE_GALLERY_CSS)
