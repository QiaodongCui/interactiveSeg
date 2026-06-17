"""
Show interactive refinement: box prompt + positive clicks on the missing dress,
+ negative clicks on background. Mimics what you'd do by hand in the ISAT GUI.
"""
import os
import numpy as np
import torch
from PIL import Image

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
from ISAT.segment_any.segment_any import SegAny

CKPT = os.path.join(os.path.dirname(__import__("ISAT").__file__), "checkpoints", "sam_hq_vit_l.pth")
SRC = "images/16757613eb01ad14b7eb31128e44dfd9401742377.jpg"

img = Image.open(SRC).convert("RGB")
W, H = img.size
arr = np.array(img)
seg = SegAny(CKPT, use_bfloat16=True)
seg.set_image(arr)

box = np.array([0.58 * W, 0.08 * H, 0.86 * W, 0.99 * H])
# positive clicks down the character's body (dress/skirt/legs), negatives on bg
pos = [(1240, 330), (1300, 430), (1360, 540), (1300, 660), (1420, 820), (1480, 980)]
neg = [(1140, 1020), (1630, 160), (1150, 120)]
pts = np.array(pos + neg, dtype=float)
lbls = np.array([1] * len(pos) + [0] * len(neg))

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
    masks, scores, logits = seg.predictor.predict(
        point_coords=pts, point_labels=lbls, box=box, multimask_output=False
    )
mask = np.asarray(masks).astype(bool)
if mask.ndim == 3:
    mask = mask[0]
print("coverage %.1f%%" % (100.0 * mask.mean()))

rgba = img.convert("RGBA")
a = np.array(rgba)
a[..., 3] = (mask * 255).astype(np.uint8)
out = Image.fromarray(a)
out.save("out/Banner_isat_refined.png")
bg = Image.new("RGBA", out.size, (255, 0, 255, 255))
comp = Image.alpha_composite(bg, out).convert("RGB")
m = max(comp.size)
if m > 1400:
    s = 1400 / m
    comp = comp.resize((int(comp.size[0] * s), int(comp.size[1] * s)))
comp.save("out/qa_Banner_isat_refined.png")
print("saved out/qa_Banner_isat_refined.png")
