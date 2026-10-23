"""Chạy SAM 3.1 (detector của checkpoint facebook/sam3.1) theo từng frame với prompt văn bản, lưu cache JSON.
CPU: rất chậm (~30 s/prompt/frame đo trên box) -> chỉ chạy mẫu nhỏ. GPU (T4): dùng cùng script với --device cuda.
HF_TOKEN phải được nạp từ môi trường (không ghi vào file)."""
import argparse, json, os, sys, time, gc, glob
import torch
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))

def load(device):
    from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf
    from sam3.model.sam3_image_processor import Sam3Processor
    ck = os.environ.get("SAM31_CKPT") or download_ckpt_from_hf(version="sam3.1")  # ưu tiên checkpoint từ private Kaggle dataset
    m = build_sam3_image_model(device=device, checkpoint_path=None, load_from_HF=False)
    sd = torch.load(ck, map_location="cpu", weights_only=True, mmap=True); sd = sd.get("model", sd)
    r = m.load_state_dict({k[9:]: v for k, v in sd.items() if k.startswith("detector.")}, strict=False)
    del sd; gc.collect()
    print(f"[sam3.1] loaded detector, missing={len(r.missing_keys)} unexpected={len(r.unexpected_keys)}", flush=True)
    return Sam3Processor(m, device=device, confidence_threshold=0.3)

def pick_dtype(device, want="auto"):
    """fp16/bf16 autocast. T4 (sm_75) không hỗ trợ bf16 thật (torch có thể báo True do giả lập) -> dùng fp16."""
    if want == "fp32": return None
    if want in ("fp16", "bf16"): return torch.float16 if want == "fp16" else torch.bfloat16
    if device == "cpu": return torch.bfloat16
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

def sync(device):
    if device.startswith("cuda"): torch.cuda.synchronize()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_dirs", nargs="+"); ap.add_argument("--out", required=True, help="file .json (1 clip) hoặc thư mục (nhiều clip -> <clip>.json)")
    ap.add_argument("--stride", type=int, default=5); ap.add_argument("--max", type=int, default=20)
    ap.add_argument("--prompts", default="motorcycle,person,head"); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    proc = load(a.device)   # nạp model 1 lần cho mọi clip
    dt = pick_dtype(a.device, a.dtype); print("[sam3.1] autocast dtype:", dt, flush=True)
    P = a.prompts.split(",")
    for fd in a.frames_dirs:
        out = a.out if a.out.endswith(".json") else os.path.join(a.out, os.path.basename(fd.rstrip("/")) + ".json")
        files = sorted(glob.glob(os.path.join(fd, "*.jpg")) + glob.glob(os.path.join(fd, "*.png")))[::a.stride][:a.max]
        cache = json.load(open(out)) if os.path.exists(out) else {}
        for j, f in enumerate(files):
            k = os.path.basename(f)
            if k in cache: continue
            im = Image.open(f).convert("RGB"); rec = {"_t_prompt": {}}
            ctx = torch.autocast(a.device.split(":")[0], dtype=dt) if dt is not None else torch.autocast(a.device.split(":")[0], enabled=False)
            with ctx, torch.inference_mode():
                sync(a.device); t = time.time(); st = proc.set_image(im); sync(a.device); rec["_t_image"] = time.time() - t
                for p in P:
                    t1 = time.time(); o = proc.set_text_prompt(state=st, prompt=p)
                    rec[p] = {"boxes": o["boxes"].float().cpu().tolist(), "scores": o["scores"].float().cpu().tolist()}
                    rec["_t_prompt"][p] = time.time() - t1
            del st
            rec["_sec"] = rec["_t_image"] + sum(rec["_t_prompt"].values()); rec["_size"] = list(im.size)
            rec["_warmup"] = j == 0; rec["_dtype"] = str(dt); cache[k] = rec
            print(os.path.basename(fd), k, {p: len(rec[p]["scores"]) for p in P}, f"img {rec['_t_image']:.2f}s", {p: round(v, 2) for p, v in rec["_t_prompt"].items()}, flush=True)
        json.dump(cache, open(out, "w"))
