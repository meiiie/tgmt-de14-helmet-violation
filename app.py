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

def analyze(image, video, backend, stride, max_sec, n_min, imgsz):
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
        pl = Pipeline(det, cls, fps=fps / stride, n_min=int(n_min)); name = "video"
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

with gr.Blocks(title="Đề 14 – Phát hiện không đội mũ bảo hiểm") as demo:
    gr.Markdown("# Đề 14 – Phát hiện người đi xe máy không đội mũ bảo hiểm\n"
                "Tải ảnh **hoặc** video. Mỗi rider (người lái + người ngồi sau) được gắn nhãn *có mũ / không mũ / chưa xác định*; "
                "với video, nhãn được bỏ phiếu theo track (ByteTrack). Ô vàng = chưa đủ bằng chứng.")
    with gr.Row():
        with gr.Column():
            img_in = gr.Image(label="Ảnh", type="pil")
            vid_in = gr.Video(label="Video")
            backend = gr.Dropdown(BACKENDS, value=BACKENDS[0], label="Backend")
            with gr.Row():
                stride = gr.Slider(1, 10, value=3, step=1, label="Lấy mẫu mỗi N frame (video)")
                max_sec = gr.Slider(1, 60, value=8, step=1, label="Tối đa (giây video)")
            with gr.Row():
                n_min = gr.Slider(1, 10, value=3, step=1, label="Số quan sát tối thiểu / track")
                imgsz = gr.Dropdown([640, 960, 1280], value=1280, label="imgsz YOLO")
            btn = gr.Button("Phân tích", variant="primary")
        with gr.Column():
            out_img = gr.Image(label="Kết quả (ảnh)")
            out_vid = gr.Video(label="Kết quả (video)")
            summary = gr.Markdown()
    table = gr.Dataframe(headers=["track_id", "thời gian", "nhãn", "tỉ lệ không mũ", "số quan sát"], label="Bảng vi phạm / theo track")
    gallery = gr.Gallery(label="Ảnh bằng chứng (crop đầu)", columns=6, height=220)
    csv_out = gr.File(label="Tải CSV")
    btn.click(analyze, [img_in, vid_in, backend, stride, max_sec, n_min, imgsz], [out_img, out_vid, summary, table, gallery, csv_out], api_name="analyze")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--share", action="store_true"); ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    demo.queue().launch(server_name="0.0.0.0", server_port=a.port, share=a.share, show_error=True)
