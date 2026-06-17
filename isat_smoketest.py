"""
End-to-end GPU smoke test of ISAT's SAM backend (no GUI):
load HQ-SAM ViT-L, box-prompt the main character out of the busy banner,
save the cutout + magenta QA. Validates the interactive pipeline on the 4090.
"""
import os
import numpy as np
from PIL import Image

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from ISAT.segment_any.segment_any import SegAny

CKPT = os.path.join(os.path.dirname(__import__("ISAT").__file__), "checkpoints", "sam_hq_vit_l.pth")
SRC = "images/16757613eb01ad14b7eb31128e44dfd9401742377.jpg"

img = Image.open(SRC).convert("RGB")
W, H = img.size
print(f"image {W}x{H}")
arr = np.array(img)

seg = SegAny(CKPT, use_bfloat16=True)
seg.set_image(arr)

# Box around the maid character (right-center of the banner). Fractions of W,H.
box = np.array([0.58 * W, 0.08 * H, 0.86 * W, 0.99 * H])
print("box:", box.astype(int).tolist())
masks = seg.predict_with_box_prompt(box)
mask = np.asarray(masks).astype(bool)
if mask.ndim == 3:
    mask = mask[0]
print("mask shape", mask.shape, "coverage %.1f%%" % (100.0 * mask.mean()))

rgba = img.convert("RGBA")
a = np.array(rgba)
a[..., 3] = (mask * 255).astype(np.uint8)
out = Image.fromarray(a)
out.save("out/Banner_isat_hqsam.png")

# magenta QA
bg = Image.new("RGBA", out.size, (255, 0, 255, 255))
comp = Image.alpha_composite(bg, out).convert("RGB")
m = max(comp.size)
if m > 1400:
    s = 1400 / m
    comp = comp.resize((int(comp.size[0] * s), int(comp.size[1] * s)))
comp.save("out/qa_Banner_isat_hqsam.png")
print("saved out/Banner_isat_hqsam.png + out/qa_Banner_isat_hqsam.png")
