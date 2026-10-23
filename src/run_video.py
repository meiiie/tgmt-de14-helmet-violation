"""Demo đầu-cuối trên 1 video hoặc thư mục video (xử lý theo lô).
python src/run_video.py <video|dir> --detector yolo|sam31 --classifier yolo|dino --stride 3 --out outputs/demo"""
import argparse, os, sys, glob, json, cv2
sys.path.insert(0, os.path.dirname(__file__))
from de14.pipeline import Pipeline

def frames(path, stride, max_frames):
    cap = cv2.VideoCapture(path); i = k = 0
    while True:
        ok, f = cap.read()
        if not ok or (max_frames and k >= max_frames): break
        if i % stride == 0:
            k += 1; yield i, f
        i += 1

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("src")
    ap.add_argument("--detector", default="yolo"); ap.add_argument("--classifier", default="yolo")
    ap.add_argument("--stride", type=int, default=3); ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=1280); ap.add_argument("--out", default="outputs/demo")
    a = ap.parse_args()
    from de14.detectors import YoloDetector, Sam3Detector
    from de14.classifiers import YoloHelmetCls, DinoProbe
    det = YoloDetector(imgsz=a.imgsz) if a.detector == "yolo" else Sam3Detector()
    cls = YoloHelmetCls() if a.classifier == "yolo" else DinoProbe.load("weights/dino_probe.joblib")
    vids = sorted(glob.glob(os.path.join(a.src, "*"))) if os.path.isdir(a.src) else [a.src]
    summary = {}
    for v in vids:
        fps = cv2.VideoCapture(v).get(cv2.CAP_PROP_FPS) or 25
        name = os.path.splitext(os.path.basename(v))[0]
        pl = Pipeline(det, cls, fps=fps / a.stride)
        # thời gian trong CSV = chỉ số frame đã lấy mẫu / fps_hiệu_dụng = giây thực trong video
        r = pl.run(((i // a.stride, f) for i, f in frames(v, a.stride, a.max_frames)), out_dir=a.out, video_name=name)
        cnt = {}
        for t in r["tracks"].values(): cnt[t["decision"]] = cnt.get(t["decision"], 0) + 1
        summary[name] = {"frames": r["frames"], "fps": round(r["fps"], 2), "tracks": cnt}
        print(name, summary[name], flush=True)
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), ensure_ascii=False, indent=1)
