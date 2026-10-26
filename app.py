"""Đề 14 – Gradio demo: phát hiện người đi xe máy không đội mũ bảo hiểm (ảnh / video).
CPU mặc định: YOLO11s (xe + người) + ByteTrack + phân loại đầu (YOLO helmet hoặc DINOv3-S probe).
GPU tuỳ chọn: SAM 3.1 (cần checkpoint gated qua HF_TOKEN hoặc SAM31_CKPT).
Chạy: python app.py   (Colab/Kaggle: python app.py --share)"""
import argparse, os, sys, csv, glob, shutil, subprocess, tempfile, time
import cv2, numpy as np, gradio as gr
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from de14.pipeline import Pipeline
from de14.core import VIOL, OK, UNDET

LBL = {VIOL: "không mũ", OK: "có mũ", UNDET: "chưa xác định"}
BACKENDS = ["YOLO11s + YOLO helmet (CPU)", "YOLO11s + DINOv3-S probe (CPU)", "SAM 3.1 + DINOv3-S probe (GPU)"]
import json as _json, torch as _torch
# Cấu hình mặc định chọn ở kernel v4 (F1 cao nhất với P tính cả track không khớp): SAM 3.1 4 prompt + DINOv3-S + bộ lọc F2
TRACK_FILTER = _json.load(open("configs/track_filter.json"))["track_filter"]
DEFAULT_BACKEND = BACKENDS[2] if (_torch.cuda.is_available() and (os.environ.get("HF_TOKEN") or os.environ.get("SAM31_CKPT"))) else BACKENDS[1]
_cache = {}

def get_models(backend, imgsz):
    key = (backend, imgsz)
    if key in _cache: return _cache[key]
    from de14.detectors import YoloDetector, Sam3Detector
    from de14.classifiers import YoloHelmetCls, DinoProbe
    det = Sam3Detector(device="cuda") if backend.startswith("SAM") else YoloDetector(imgsz=imgsz, conf=0.2)
    cls = YoloHelmetCls() if "YOLO helmet" in backend else DinoProbe.load("weights/dino_probe.joblib")
    _cache[key] = (det, cls); return det, cls

def _to_h264(src):
    dst = src.replace(".mp4", "_h264.mp4")
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst])
    return dst if r.returncode == 0 else src

def analyze(image, video, backend, stride, max_sec, n_min, imgsz, use_filter=True):
    if image is None and video is None:
        raise gr.Error("Hãy tải lên ảnh hoặc video")
    det, cls = get_models(backend, int(imgsz))
    out = tempfile.mkdtemp(prefix="de14_")
    t0 = time.time()
    if video is not None:
        cap = cv2.VideoCapture(video); fps = cap.get(cv2.CAP_PROP_FPS) or 25; stride = int(stride)
        maxf = int(max_sec * fps)
        def gen():
            i = 0
            while i < maxf:
                ok, f = cap.read()
                if not ok: break
                if i % stride == 0: yield i // stride, f
                i += 1
        pl = Pipeline(det, cls, fps=fps / stride, n_min=int(n_min), track_filter=TRACK_FILTER if use_filter else None); name = "video"
        r = pl.run(gen(), out_dir=out, video_name=name)
    else:
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        pl = Pipeline(det, cls, fps=1, n_min=1); name = "image"
        r = pl.run(iter([(0, img)]), out_dir=out, video_name=name)
    el = time.time() - t0
    rows, gallery = [], []
    csv_path = os.path.join(out, f"{name}_tracks.csv")
    for row in csv.DictReader(open(csv_path, encoding="utf-8")):
        conf = "" if row["no_helmet_ratio"] in ("", "None") else f"{float(row['no_helmet_ratio']):.2f}"
        rows.append([int(row["track_id"]), f"{float(row['t_start']):.1f}–{float(row['t_end']):.1f}s", LBL.get(row["decision"], row["decision"]), conf, int(row["n_obs"])])
        if row["evidence"]:
            gallery.append((os.path.join(out, row["evidence"]), f"#{row['track_id']} {LBL.get(row['decision'])} t={float(row['t_start']):.1f}s"))
    rows.sort(key=lambda x: (x[2] != "không mũ", x[0]))
    vis_img, vis_vid = None, None
    if video is not None: vis_vid = _to_h264(os.path.join(out, f"{name}_annotated.mp4"))
    elif pl.last_vis is not None: vis_img = cv2.cvtColor(pl.last_vis, cv2.COLOR_BGR2RGB)
    n_v = sum(1 for x in rows if x[2] == "không mũ"); n_u = sum(1 for x in rows if x[2] == "chưa xác định")
    summary = (f"**{len(rows)} xe/track** · không mũ: **{n_v}** · chưa xác định: {n_u} · {r['frames']} frame · "
               f"{el:.1f}s ({r['frames']/max(el,1e-6):.2f} FPS) · backend: {backend}")
    return vis_img, vis_vid, summary, rows, gallery, csv_path

# ---------------- Realtime ----------------
RT_DEFAULT = {"imgsz": 640, "every_n": 5}   # cập nhật theo kết quả đo kernel v6 (README mục G)
def _rt_pipeline(w, sz, n, f2):
    from de14.realtime import RealtimePipeline
    return RealtimePipeline(f"weights/{w}.pt", "weights/dino_probe.joblib", imgsz=int(sz), every_n=int(n),
                            track_filter=TRACK_FILTER if f2 else None, fps=25)

def rt_stream(src, file, w, sz, n, f2):
    path = (src or "").strip() or file
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

with gr.Blocks(title="Đề 14 – Phát hiện không đội mũ bảo hiểm") as demo:
    gr.Markdown("# Đề 14 – Phát hiện người đi xe máy không đội mũ bảo hiểm\n"
                "Tải ảnh **hoặc** video. Mỗi rider (người lái + người ngồi sau) được gắn nhãn *có mũ / không mũ / chưa xác định*; "
                "với video, nhãn được bỏ phiếu theo track (ByteTrack). Ô vàng = chưa đủ bằng chứng.")
    with gr.Tab("Offline (ảnh / video, SAM 3.1 hoặc YOLO)"):
        with gr.Row():
            with gr.Column():
                img_in = gr.Image(label="Ảnh", type="pil")
                vid_in = gr.Video(label="Video")
                backend = gr.Dropdown(BACKENDS, value=DEFAULT_BACKEND, label="Backend")
                with gr.Row():
                    stride = gr.Slider(1, 10, value=3, step=1, label="Lấy mẫu mỗi N frame (video)")
                    max_sec = gr.Slider(1, 60, value=8, step=1, label="Tối đa (giây video)")
                with gr.Row():
                    n_min = gr.Slider(1, 10, value=3, step=1, label="Số quan sát tối thiểu / track")
                    imgsz = gr.Dropdown([640, 960, 1280], value=1280, label="imgsz YOLO")
                    use_filter = gr.Checkbox(value=True, label="Lọc F2: bỏ track gần đứng yên không có người (giữ rider dừng đèn đỏ)")
                btn = gr.Button("Phân tích", variant="primary")
            with gr.Column():
                out_img = gr.Image(label="Kết quả (ảnh)")
                out_vid = gr.Video(label="Kết quả (video)")
                summary = gr.Markdown()
        table = gr.Dataframe(headers=["track_id", "thời gian", "nhãn", "tỉ lệ không mũ", "số quan sát"], label="Bảng vi phạm / theo track")
        gallery = gr.Gallery(label="Ảnh bằng chứng (crop đầu)", columns=6, height=220)
        csv_out = gr.File(label="Tải CSV")
        btn.click(analyze, [img_in, vid_in, backend, stride, max_sec, n_min, imgsz, use_filter], [out_img, out_vid, summary, table, gallery, csv_out], api_name="analyze")

    with gr.Tab("Realtime (webcam / RTSP / file)"):
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
    ap = argparse.ArgumentParser(); ap.add_argument("--share", action="store_true"); ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    demo.queue().launch(server_name="0.0.0.0", server_port=a.port, share=a.share, show_error=True)
