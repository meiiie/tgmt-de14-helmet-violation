# Đề 14 – Kaggle GPU kernel (PRIVATE). HF_TOKEN chỉ lấy từ Kaggle Secrets (UserSecretsClient), không nhúng vào file.
import os, subprocess, glob, json, sys, time, shutil
import signal, threading
T0 = time.time(); BUDGET = float(os.environ.get("DE14_BUDGET_SEC", 100 * 60))   # HARD timeout toàn kernel: 100 phút (quy tắc chia GPU)
def time_left(): return BUDGET - (time.time() - T0)
_procs = []
def sh(c, check=False):
    """Chạy lệnh trong process group riêng; bị giết (cả cây) khi hết ngân sách toàn kernel."""
    print("$", c, flush=True)
    if time_left() < 30: print("SKIP (hard timeout)", c, flush=True); return 124
    p = subprocess.Popen(c, shell=True, start_new_session=True); _procs.append(p)
    try: rc = p.wait(timeout=max(1, time_left() - 20))
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL); p.wait(); rc = 124; print("KILLED (hard timeout)", c, flush=True)
    if check and rc: raise SystemExit(f"failed: {c}")
    return rc
def _finalize(reason):
    """Ghi kết quả từng phần + dọn trọng số gated khỏi output."""
    W_ = "/kaggle/working"
    subprocess.run(f"mkdir -p {W_}/results && cp -r {W_}/de14/outputs/metrics {W_}/results/ 2>/dev/null; cp {W_}/run_log.json {W_}/results/ 2>/dev/null; "
                   f"echo '{reason}' > {W_}/results/STATUS.txt; rm -rf {W_}/frames {W_}/sam3.1_multiplex.pt {W_}/dinov3_local {W_}/de14/weights/*.engine {W_}/de14/weights/*.onnx", shell=True)
def _watchdog():
    while time_left() > 0: time.sleep(5)
    print("HARD TIMEOUT 100 min: killing all steps, writing partial results", flush=True)
    for p in _procs:
        try: os.killpg(p.pid, signal.SIGKILL)
        except Exception: pass
    _finalize("partial: hard timeout at 100 min"); os._exit(0)
threading.Thread(target=_watchdog, daemon=True).start()
sh("nvidia-smi")
# 1) checkpoint từ private dataset holilihu/de7-weights (SAM 3.1 chia part + sha256; DINOv3 S/B tên phẳng "<repo>__<file>")
import hashlib
ck, dl = [], []
p0 = glob.glob("/kaggle/input/**/sam31_part_00", recursive=True)
if p0:
    wd = os.path.dirname(p0[0]); out = "/kaggle/working/sam3.1_multiplex.pt"
    parts = sorted(glob.glob(f"{wd}/sam31_part_*")); h = hashlib.sha256()
    with open(out, "wb") as fo:
        for pp in parts:
            with open(pp, "rb") as fi:
                while True:
                    b_ = fi.read(1 << 24)
                    if not b_: break
                    fo.write(b_); h.update(b_)
    want = open(f"{wd}/sam31_sha256.txt").read().split()[0].strip()
    if h.hexdigest() == want:
        ck = [out]; os.environ["SAM31_CKPT"] = out; print(f"SAM 3.1 reassembled from {len(parts)} parts, sha256 OK", flush=True)
    else:
        print("ERROR: SAM 3.1 sha256 mismatch", h.hexdigest(), "!=", want, flush=True); os.remove(out)
    DL = "/kaggle/working/dinov3_local"
    for f in glob.glob(f"{wd}/dinov3-*__*"):
        rep, fn = os.path.basename(f).split("__", 1); os.makedirs(f"{DL}/{rep}", exist_ok=True)
        dst = f"{DL}/{rep}/{fn}"
        if not os.path.exists(dst): os.symlink(f, dst)
    # kiểm sha256 cho DINOv3 (file <repo>__sha256.txt); sai -> bỏ model đó (sẽ báo 'chưa xác định', KHÔNG fallback DINOv2)
    for shf in glob.glob(f"{wd}/dinov3-*__sha256.txt"):
        rep = os.path.basename(shf).split("__")[0]; want_ = open(shf).read().split()[0]; h2 = hashlib.sha256()
        with open(f"{wd}/{rep}__model.safetensors", "rb") as fi:
            for b_ in iter(lambda: fi.read(1 << 24), b""): h2.update(b_)
        ok = h2.hexdigest() == want_; print(f"DINOv3 {rep} sha256", "OK" if ok else "MISMATCH", flush=True)
        if not ok: shutil.rmtree(f"{DL}/{rep}", ignore_errors=True)
    if os.path.isdir(DL): dl = [DL]; os.environ["DINO_LOCAL_DIR"] = DL; print("DINOv3 local:", sorted(os.listdir(DL)), flush=True)
# 2) không dùng Kaggle Secret: mọi trọng số gated lấy từ private dataset holilihu/de7-weights
HAS_TOKEN = bool(ck and dl); os.environ.pop("HF_TOKEN", None)
W = "/kaggle/working"; os.chdir(W)
sh("git clone -q https://github.com/meiiie/tgmt-de14-helmet-violation.git de14", check=True); os.chdir(f"{W}/de14")
sh("git clone -q https://github.com/facebookresearch/sam3.git third_party_sam3 && cd third_party_sam3 && git apply ../patches/sam3_cpu.patch && pip -q install --no-deps -e .")
sh("pip -q install ultralytics timm ftfy regex iopath einops pycocotools lap 'transformers>=4.56' scikit-learn joblib")
sys.path.insert(0, f"{W}/de14/third_party_sam3")
# ---- dữ liệu: dataset input hozngvan/helmet-detection (mirror của HELMET, OSF CC-BY 4.0) ----
roots = glob.glob("/kaggle/input/*")
print("inputs", roots)
split = (glob.glob("/kaggle/input/**/data_split.csv", recursive=True) or [None])[0]
ann = os.path.dirname((glob.glob("/kaggle/input/**/Bago_urban_3.csv", recursive=True) or [""])[0])
cand = [d for d in glob.glob("/kaggle/input/**/Bago_urban_3", recursive=True) if os.path.isdir(d)]
print("split", split, "ann", ann, "frame dirs", cand, flush=True)
fdir = os.path.dirname(cand[0]); os.makedirs("data/helmet", exist_ok=True); shutil.copy(split, "data/helmet/data_split.csv")
ex = sorted(os.listdir(cand[0]))[:3]; print("example frame names", ex, flush=True)
# chuẩn hoá tên frame -> NNN.jpg qua symlink
FR = f"{W}/frames"
TEST = ["Bago_highway_10", "Bago_highway_20", "Bago_rural_3", "Bago_rural_49", "Bago_urban_3", "Bago_urban_50",
        "Mandalay_1_105", "Mandalay_2_10", "Naypyitaw_1_10", "Yangon_II_10"]
import csv as _csv
sp = {r["video_id"]: r["Set"] for r in _csv.DictReader(open(split))}
# layout mirror (v1 log): /kaggle/input/helmet-detection/<split_dir>/<clip>/images/*.jpg
CLIPDIR = {}
for d in glob.glob("/kaggle/input/helmet-detection/*/*/images"):
    c = d.split("/")[-2]; CLIPDIR.setdefault(c, d)
print("clips with images:", len(CLIPDIR), "split dirs:", sorted({d.split('/')[-3] for d in glob.glob('/kaggle/input/helmet-detection/*/*/images')}), flush=True)
TEST = [c for c in TEST if sp.get(c) == "test" and c in CLIPDIR]
if len(TEST) < 6: TEST += [c for c, s in sp.items() if s == "test" and c not in TEST and c in CLIPDIR][:6 - len(TEST)]
TRAIN = [c for c, s in sp.items() if s == "training" and c in CLIPDIR][::4][:40]
open("data/helmet/train_clips.txt", "w").write("\n".join(TRAIN))
import re
for c in TEST + TRAIN:
    os.makedirs(f"{FR}/{c}", exist_ok=True)
    fs = sorted(f for f in os.listdir(CLIPDIR[c]) if f.lower().endswith((".jpg", ".png")))
    for i, f in enumerate(fs):
        m = re.findall(r"\d+", f); n = int(m[-1]) if m else i + 1
        dst = f"{FR}/{c}/{n:03d}.jpg"
        if not os.path.exists(dst): os.symlink(f"{CLIPDIR[c]}/{f}", dst)
os.environ.update(HELMET_FRAMES=FR, HELMET_ANN=ann)
print("TEST", TEST, "TRAIN", len(TRAIN), flush=True)
sh("mkdir -p weights && wget -q -O weights/yolo11s.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt")
sh("python -c \"from huggingface_hub import hf_hub_download as h;import shutil;shutil.copy(h('nnsohamnn/helmet-detection-yolo11','yolov11s(80 epochs).pt'),'weights/helmet_yolo11s_nnsohamnn.pt')\"")
T = " ".join(TEST); log = {}
def timed(name, cmd):
    if time_left() < 300:
        print("SKIP (time budget)", name, flush=True); log[name] = {"rc": "skipped_budget"}; return
    t = time.time(); rc = sh(cmd); log[name] = {"rc": rc, "sec": round(time.time() - t, 1)}
    json.dump(log, open(f"{W}/run_log.json", "w"), indent=1)
# ---- v8: thử tầng HEAD DETECTOR (DEIMv2-Wholebody49 ONNX, PINTO0309, Apache-2.0) trong crop rider, so với baseline realtime ----
sh("wget -q -O weights/yolo26s.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt")
REL = "https://github.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation/releases/download/weights"
HD = ["deimv2_hgnetv2_n_wholebody49_boxes_only", "deimv2_dinov3_s_wholebody49_boxes_only"]
for h in HD: sh(f"wget -q -O weights/{h}.onnx {REL}/{h}.onnx")
timed("ort_install", "timeout -k 10 600 pip -q install onnxruntime-gpu")
sh("python -c 'import onnxruntime as o,ultralytics,torch;print(\"ort\",o.__version__,o.get_available_providers(),\"ultralytics\",ultralytics.__version__,\"torch\",torch.__version__)'")
os.environ["CROP_STRIDE"] = "5"; timed("crops", "python -u src/build_head_crops.py")
if HAS_TOKEN:
    timed("probe_vits16", "python -u src/train_probe.py facebook/dinov3-vits16-pretrain-lvd1689m")
    PR = "weights/dino_probe_dinov3-vits16-pretrain-lvd1689m.joblib"
    timed("rt_base", f"timeout -k 10 600 python -u src/eval_realtime.py --weights weights/yolo26s.pt --imgsz 640 --every_n 5 --probe {PR} --tag rt_yolo26s_640_n5")
    for h in HD:
        tg = "hgn" if "hgnetv2" in h else "dv3s"
        timed(f"rt_head_{tg}", f"timeout -k 10 900 python -u src/eval_realtime.py --weights weights/yolo26s.pt --imgsz 640 --every_n 5 --probe {PR} --head_onnx weights/{h}.onnx --tag rt_yolo26s_640_n5_head_{tg}")
    sh(f"mkdir -p {W}/results && cp {PR} {W}/results/ 2>/dev/null")
else:
    print("SKIP gated parts (SAM 3.1, DINOv3): HF_TOKEN secret missing", flush=True)
_finalize("complete")  # ghi kết quả + xoá trọng số gated / engine khỏi output
sh(f"rm -rf {W}/de14/third_party_sam3/.git")
print(json.dumps(log, indent=1))
