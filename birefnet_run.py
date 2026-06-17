"""
Run BiRefNet matting checkpoints + the ToonOut anime fine-tune on an image and
export a soft-alpha RGBA cutout plus magenta QA previews (full + hair crop).

Usage:
    python birefnet_run.py <input_image> [out_dir]

Models tried (each independent; failures are skipped, not fatal):
    - ZhengPeng7/BiRefNet_HR-matting   (2048px, best general hair detail, soft alpha)
    - ZhengPeng7/BiRefNet-matting      (1024px, soft alpha)
    - joelseytre/toonout               (anime fine-tune of BiRefNet)
"""
import sys, os, traceback
import torch
from torchvision import transforms
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = DEVICE == "cuda"

MODELS = [
    {"name": "birefnet_hr_matting", "repo": "ZhengPeng7/BiRefNet_HR-matting", "res": 2048},
    {"name": "birefnet_matting",    "repo": "ZhengPeng7/BiRefNet-matting",    "res": 1024},
    {"name": "toonout",             "repo": "joelseytre/toonout",             "res": 1024,
     "fallback_arch": "ZhengPeng7/BiRefNet"},
]


def load_model(spec):
    from transformers import AutoModelForImageSegmentation
    try:
        m = AutoModelForImageSegmentation.from_pretrained(spec["repo"], trust_remote_code=True)
        print(f"  loaded {spec['repo']} via AutoModel")
        return m
    except Exception as e:
        print(f"  AutoModel load failed for {spec['repo']}: {e}")
        if "fallback_arch" not in spec:
            raise
    # Fallback: load base architecture, then graft the fine-tuned weights.
    from huggingface_hub import HfApi, hf_hub_download
    from safetensors.torch import load_file
    print(f"  falling back: base arch {spec['fallback_arch']} + weights from {spec['repo']}")
    m = AutoModelForImageSegmentation.from_pretrained(spec["fallback_arch"], trust_remote_code=True)
    files = HfApi().list_repo_files(spec["repo"])
    weight_file = None
    for cand in files:
        if cand.endswith((".safetensors", ".pth", ".pt", ".bin")):
            weight_file = cand
            break
    if weight_file is None:
        raise RuntimeError(f"no weight file found in {spec['repo']}: {files}")
    print(f"  weight file: {weight_file}")
    path = hf_hub_download(spec["repo"], weight_file)
    sd = load_file(path) if weight_file.endswith(".safetensors") else torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    def clean_key(k):
        # strip DDP / torch.compile wrappers: module._orig_mod.<...>
        for p in ("module.", "_orig_mod."):
            if k.startswith(p):
                k = k[len(p):]
        return k

    sd = {clean_key(k): v for k, v in sd.items()}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print(f"  load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    return m


def run(model, image, res):
    tf = transforms.Compose([
        transforms.Resize((res, res)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    x = tf(image).unsqueeze(0).to(DEVICE)
    if USE_FP16:
        x = x.half()
    with torch.no_grad():
        out = model(x)
        pred = out[-1] if isinstance(out, (list, tuple)) else out
        pred = pred.sigmoid().float().cpu()[0].squeeze()
    mask = transforms.ToPILImage()(pred).resize(image.size)
    rgba = image.convert("RGBA")
    rgba.putalpha(mask)
    return rgba


def qa(rgba, base):
    bg = Image.new("RGBA", rgba.size, (255, 0, 255, 255))
    comp = Image.alpha_composite(bg, rgba).convert("RGB")
    w, h = comp.size
    m = max(w, h)
    full = comp.resize((int(w * 1400 / m), int(h * 1400 / m))) if m > 1400 else comp
    full.save(base + "_qa.png")
    # hair crop (tuned for portrait character cards; adjust if needed)
    c = rgba.crop((int(w * 0.18), int(h * 0.06), int(w * 0.92), int(h * 0.40)))
    cbg = Image.new("RGBA", c.size, (255, 0, 255, 255))
    cc = Image.alpha_composite(cbg, c).convert("RGB")
    cc = cc.resize((cc.size[0] * 2, cc.size[1] * 2), Image.NEAREST)
    cc.save(base + "_qa_hair.png")


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "images/Sandrone_Card.png"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "out"
    os.makedirs(out_dir, exist_ok=True)
    print(f"device={DEVICE} fp16={USE_FP16}  input={src}")
    image = Image.open(src).convert("RGB")
    stem = os.path.splitext(os.path.basename(src))[0]
    for spec in MODELS:
        print(f"\n=== {spec['name']} ({spec['repo']}) @ {spec['res']}px ===")
        try:
            model = load_model(spec).to(DEVICE).eval()
            if USE_FP16:
                model = model.half()
            import time
            t0 = time.time()
            rgba = run(model, image, spec["res"])
            print(f"  inference {time.time() - t0:.1f}s")
            base = os.path.join(out_dir, f"{stem}__{spec['name']}")
            rgba.save(base + ".png")
            qa(rgba, base)
            print(f"  saved {base}.png (+ _qa, _qa_hair)")
            del model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            print(f"  FAILED {spec['name']}:")
            traceback.print_exc()


if __name__ == "__main__":
    main()
