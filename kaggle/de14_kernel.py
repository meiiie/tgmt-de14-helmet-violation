# Đề 14 – Kaggle GPU kernel (PRIVATE). HF_TOKEN chỉ lấy từ Kaggle Secrets (UserSecretsClient), không nhúng vào file.
import os, subprocess, glob, json, sys, time, shutil
def sh(c, check=False):
    print("$", c, flush=True); r = subprocess.run(c, shell=True)
    if check and r.returncode: raise SystemExit(f"failed: {c}")
    return r.returncode
sh("nvidia-smi")
T0 = time.time(); BUDGET = float(os.environ.get("DE14_BUDGET_SEC", 110 * 60))   # giới hạn ~2h (quy tắc chia GPU)
def time_left(): return BUDGET - (time.time() - T0)
# 1) checkpoint fallback: private Kaggle dataset đính kèm (nếu có) -> không cần token
ck = glob.glob("/kaggle/input/**/sam3.1_multiplex.pt", recursive=True)
if ck: os.environ["SAM31_CKPT"] = ck[0]; print("SAM 3.1 ckpt from attached private dataset", flush=True)
dl = [os.path.dirname(d) for d in glob.glob("/kaggle/input/**/dinov3-vit*16-pretrain-lvd1689m", recursive=True)]
if dl: os.environ["DINO_LOCAL_DIR"] = dl[0]; print("DINOv3 from attached private dataset", flush=True)
# 2) HF_TOKEN chỉ từ Kaggle Secrets
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN: available from Kaggle Secrets", flush=True)
except Exception as e:
    print("ERROR: Kaggle Secret 'HF_TOKEN' not available to this kernel (Add-ons > Secrets > HF_TOKEN, attach to kernel). "
          f"Gated SAM 3.1/DINOv3 parts will be skipped unless checkpoints are attached. ({type(e).__name__})", flush=True)
HAS_TOKEN = bool(os.environ.get("HF_TOKEN")) or bool(ck and dl)
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
TEST = [c for c in TEST if sp.get(c) == "test" and os.path.isdir(f"{fdir}/{c}")]
if len(TEST) < 6: TEST += [c for c, s in sp.items() if s == "test" and c not in TEST and os.path.isdir(f"{fdir}/{c}")][:6 - len(TEST)]
TRAIN = [c for c, s in sp.items() if s == "training" and os.path.isdir(f"{fdir}/{c}")][::4][:40]
open("data/helmet/train_clips.txt", "w").write("\n".join(TRAIN))
import re
for c in TEST + TRAIN:
    os.makedirs(f"{FR}/{c}", exist_ok=True)
    fs = sorted(f for f in os.listdir(f"{fdir}/{c}") if f.lower().endswith((".jpg", ".png")))
    for i, f in enumerate(fs):
        m = re.findall(r"\d+", f); n = int(m[-1]) if m else i + 1
        dst = f"{FR}/{c}/{n:03d}.jpg"
        if not os.path.exists(dst): os.symlink(f"{fdir}/{c}/{f}", dst)
os.environ.update(HELMET_FRAMES=FR, HELMET_ANN=ann)
print("TEST", TEST, "TRAIN", len(TRAIN), flush=True)
sh("mkdir -p weights && wget -q -O weights/yolo11s.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt")
sh("python -c \"from huggingface_hub import hf_hub_download as h;import shutil;shutil.copy(h('nnsohamnn/helmet-detection-yolo11','yolov11s(80 epochs).pt'),'weights/helmet_yolo11s_nnsohamnn.pt')\"")
T = " ".join(TEST); log = {}
def timed(name, cmd):
    if time_left() < 300:
        print("SKIP (time budget)", name, flush=True); log[name] = {"rc": "skipped_budget"}; return
    t = time.time(); rc = sh(f"timeout {int(time_left())} {cmd}"); log[name] = {"rc": rc, "sec": round(time.time() - t, 1)}
    json.dump(log, open(f"{W}/run_log.json", "w"), indent=1)
# ---- baselines (không cần token) ----
timed("b0", f"python -u src/eval_helmet.py --method b0 --clips {T} --stride 2 --tag b0_full")
timed("b1_yolo_yolohelmet", f"python -u src/eval_helmet.py --method b1 --detector yolo --classifier yolo --clips {T} --stride 2 --tag b1_yolo_yolohelmet_full --out {W}/eval_b1")
# ---- crop đầu full-res + DINOv3 S/B/L probes ----
timed("crops", "CROP_STRIDE=5 python -u src/build_head_crops.py")
if HAS_TOKEN:
    for m in ["facebook/dinov3-vits16-pretrain-lvd1689m", "facebook/dinov3-vitb16-pretrain-lvd1689m", "facebook/dinov3-vitl16-pretrain-lvd1689m"]:
        timed("probe_" + m.split("/")[-1], f"python -u src/train_probe.py {m}")
    best = max(glob.glob("outputs/metrics/probe_cv_*.json"), key=lambda f: json.load(open(f))["dino_linear"]["f1"])
    bm = json.load(open(best))["model"].split("/")[-1]; print("best probe", bm, flush=True)
    timed("b1_yolo_dinov3", f"python -u src/eval_helmet.py --method b1 --detector yolo --classifier dino --probe weights/dino_probe_{bm}.joblib --clips {T} --stride 2 --tag b1_yolo_{bm} --out {W}/eval_b1d")
    # ---- core: SAM 3.1 (motorcycle, person, head, helmet) ----
    os.makedirs("outputs/sam31", exist_ok=True)
    for c in TEST:
        timed("sam31_" + c, f"python -u src/sam31_cache.py {FR}/{c} --out outputs/sam31/{c}.json --stride 2 --max 50 --prompts motorcycle,person,head,helmet --device cuda")
    timed("core", f"python -u src/eval_helmet.py --method b1 --detector sam31 --classifier dino --probe weights/dino_probe_{bm}.joblib --clips {T} --stride 2 --tag core_sam31_{bm} --out {W}/eval_core")
    timed("core_yolohelmet", f"python -u src/eval_helmet.py --method b1 --detector sam31 --classifier yolo --clips {T} --stride 2 --tag core_sam31_yolohelmet --out {W}/eval_core2")
else:
    print("SKIP gated parts (SAM 3.1, DINOv3): HF_TOKEN secret missing", flush=True)
sh(f"mkdir -p {W}/results && cp -r outputs/metrics {W}/results/ && cp -r outputs/sam31 {W}/results/ 2>/dev/null; cp {W}/run_log.json {W}/results/")
sh(f"rm -rf {FR} {W}/de14/third_party_sam3/.git")
print(json.dumps(log, indent=1))
