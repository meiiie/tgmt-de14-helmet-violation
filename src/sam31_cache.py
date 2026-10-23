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
    ck = download_ckpt_from_hf(version="sam3.1")
    m = build_sam3_image_model(device=device, checkpoint_path=None, load_from_HF=False)
    sd = torch.load(ck, map_location="cpu", weights_only=True, mmap=True); sd = sd.get("model", sd)
    r = m.load_state_dict({k[9:]: v for k, v in sd.items() if k.startswith("detector.")}, strict=False)
    del sd; gc.collect()
    print(f"[sam3.1] loaded detector, missing={len(r.missing_keys)} unexpected={len(r.unexpected_keys)}", flush=True)
    return Sam3Processor(m, device=device, confidence_threshold=0.3)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_dir"); ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=5); ap.add_argument("--max", type=int, default=20)
    ap.add_argument("--prompts", default="motorcycle,person,head"); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    proc = load(a.device)
    files = sorted(glob.glob(os.path.join(a.frames_dir, "*.jpg")) + glob.glob(os.path.join(a.frames_dir, "*.png")))[::a.stride][:a.max]
    cache = json.load(open(a.out)) if os.path.exists(a.out) else {}
    dt = torch.bfloat16 if (a.device == "cpu" or torch.cuda.is_bf16_supported()) else torch.float16
    for f in files:
        k = os.path.basename(f)
        if k in cache: continue
        im = Image.open(f).convert("RGB"); t = time.time(); rec = {}
        with torch.autocast(a.device, dtype=dt), torch.inference_mode():
            st = proc.set_image(im)
            for p in a.prompts.split(","):
                o = proc.set_text_prompt(state=st, prompt=p)
                rec[p] = {"boxes": o["boxes"].float().cpu().tolist(), "scores": o["scores"].float().cpu().tolist()}
        del st; gc.collect()
        rec["_sec"] = time.time() - t; rec["_size"] = list(im.size); cache[k] = rec
        json.dump(cache, open(a.out, "w"))
        print(k, {p: len(rec[p]["scores"]) for p in a.prompts.split(",")}, f"{rec['_sec']:.1f}s", flush=True)
