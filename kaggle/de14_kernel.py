# Đề 14 – Kaggle GPU kernel (PRIVATE). HF_TOKEN chỉ lấy từ Kaggle Secrets (UserSecretsClient), không nhúng vào file.
import os, subprocess, glob, json, sys, time, shutil
def sh(c, check=False):
    print("$", c, flush=True); r = subprocess.run(c, shell=True)
    if check and r.returncode: raise SystemExit(f"failed: {c}")
    return r.returncode
sh("nvidia-smi")
T0 = time.time(); BUDGET = float(os.environ.get("DE14_BUDGET_SEC", 110 * 60))   # giới hạn ~2h (quy tắc chia GPU)
def time_left(): return BUDGET - (time.time() - T0)
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
    t = time.time(); rc = sh(f"timeout {int(time_left())} {cmd}"); log[name] = {"rc": rc, "sec": round(time.time() - t, 1)}
    json.dump(log, open(f"{W}/run_log.json", "w"), indent=1)
# ---- baselines (không cần token) ----
timed("b0", f"python -u src/eval_helmet.py --method b0 --clips {T} --stride 2 --tag b0_full")
# B1 (YOLO+YOLO-helmet, YOLO+DINOv3-B) đã có số ở v2 -> không chạy lại để dành GPU cho SAM fp16 + ablation
# ---- crop đầu full-res + DINOv3 S/B/L probes ----
os.environ["CROP_STRIDE"] = "5"; timed("crops", "python -u src/build_head_crops.py")
if HAS_TOKEN:
    for m in ["facebook/dinov3-vits16-pretrain-lvd1689m", "facebook/dinov3-vitb16-pretrain-lvd1689m", "facebook/dinov3-vitl16-pretrain-lvd1689m"]:
        timed("probe_" + m.split("/")[-1], f"python -u src/train_probe.py {m}")
        if log.get("probe_" + m.split("/")[-1], {}).get("rc") != 0:
            os.makedirs("outputs/metrics", exist_ok=True)
            json.dump({"model": m, "status": "chưa xác định", "reason": f"load/train failed rc={log['probe_' + m.split('/')[-1]]}"},
                      open(f"outputs/metrics/probe_cv_{m.split('/')[-1]}.json", "w"), ensure_ascii=False, indent=1)
    # ---- core: SAM 3.1 fp16 autocast (T4 không có bf16), 4 prompt; model nạp 1 lần cho mọi clip; đo thời gian set_image + từng prompt ----
    os.makedirs("outputs/sam31", exist_ok=True)
    timed("sam31_fp16_4p", f"python -u src/sam31_cache.py {' '.join(FR + '/' + c for c in TEST)} --out outputs/sam31 --stride 2 --max 50 --prompts motorcycle,person,head,helmet --dtype auto --device cuda")
    # kiểm chứng FPS 2 prompt bằng một lần chạy thật riêng (1 clip, 20 frame)
    os.makedirs("outputs/sam31_2p", exist_ok=True)
    timed("sam31_fp16_2p_timing", f"python -u src/sam31_cache.py {FR}/{TEST[0]} --out outputs/sam31_2p --stride 2 --max 20 --prompts motorcycle,person --dtype auto --device cuda")
    for bm in ["dinov3-vitb16-pretrain-lvd1689m", "dinov3-vits16-pretrain-lvd1689m", "dinov3-vitl16-pretrain-lvd1689m"]:
        if not os.path.exists(f"weights/dino_probe_{bm}.joblib"): continue
        for tagp, pr_ in [("4p", "motorcycle,person,head,helmet"), ("2p", "motorcycle,person")]:
            timed(f"core_{tagp}_{bm}", f"python -u src/eval_helmet.py --method b1 --detector sam31 --sam_prompts {pr_} --classifier dino --probe weights/dino_probe_{bm}.joblib --clips {T} --stride 2 --tag core_fp16_{tagp}_{bm} --out {W}/eval_core_{tagp}_{bm.split('-')[1]}")
else:
    print("SKIP gated parts (SAM 3.1, DINOv3): HF_TOKEN secret missing", flush=True)
sh(f"mkdir -p {W}/results && cp -r outputs/metrics {W}/results/ && cp -r outputs/sam31 outputs/sam31_2p {W}/results/ 2>/dev/null; cp {W}/run_log.json {W}/results/")
sh(f"rm -rf {FR} {W}/de14/third_party_sam3/.git /kaggle/working/sam3.1_multiplex.pt /kaggle/working/dinov3_local")  # không để trọng số gated trong output
print(json.dumps(log, indent=1))
