"""
SegPick - minimal interactive character-segmentation tool.

Backend: reuses ISAT's SegAny (HQ-SAM on GPU).
Frontend: minimal web UI (static/index.html) - left-click add, right-click remove, drag box.
Control: REST API so a human (browser) AND Claude (HTTP) can both drive it.

Run:  .venv-isat\Scripts\python.exe segtool\app.py --port 8765
Then open http://127.0.0.1:8765
"""
import os
import io
import threading

import numpy as np
import torch
from flask import Flask, request, jsonify, send_from_directory, send_file, abort
from PIL import Image

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
from ISAT.segment_any.segment_any import SegAny

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "out")
os.makedirs(OUT_DIR, exist_ok=True)

ISAT_DIR = os.path.dirname(__import__("ISAT").__file__)
DEFAULT_CKPT = os.path.join(ISAT_DIR, "checkpoints", "sam_hq_vit_l.pth")

app = Flask(__name__, static_folder=os.path.join(HERE, "static"), static_url_path="")
lock = threading.Lock()


class State:
    def __init__(self):
        self.version = 0
        self.image_path = None
        self.image = None       # np uint8 RGB
        self.size = [0, 0]      # [W, H]
        self.points = []        # [[x, y, label], ...]  label 1=add 0=remove
        self.box = None         # [x0, y0, x1, y1]
        self.mask = None        # np bool HxW


S = State()

print("Loading SAM:", DEFAULT_CKPT)
SEG = SegAny(DEFAULT_CKPT, use_bfloat16=True)
print("SAM ready on", SEG.device)


def recompute():
    """Run SAM with the current prompts -> S.mask."""
    if S.image is None or (not S.points and S.box is None):
        S.mask = None
        return
    pts = np.array([[p[0], p[1]] for p in S.points], dtype=float) if S.points else None
    lbls = np.array([p[2] for p in S.points], dtype=int) if S.points else None
    box = np.array(S.box, dtype=float) if S.box is not None else None
    with torch.inference_mode(), torch.autocast(
        SEG.device, dtype=SEG.model_dtype, enabled=torch.cuda.is_available()
    ):
        masks, scores, logits = SEG.predictor.predict(
            point_coords=pts, point_labels=lbls, box=box, multimask_output=False
        )
    m = np.asarray(masks)
    if m.ndim == 3:
        m = m[0]
    S.mask = m.astype(bool)


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/state")
def api_state():
    return jsonify({
        "version": S.version,
        "image_path": S.image_path,
        "size": S.size,
        "points": S.points,
        "box": S.box,
        "has_mask": S.mask is not None,
        "coverage": (round(100.0 * float(S.mask.mean()), 2) if S.mask is not None else 0),
        "model": os.path.basename(DEFAULT_CKPT),
        "device": SEG.device,
    })


def _png(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/api/image.png")
def api_image():
    if S.image is None:
        abort(404)
    return _png(Image.fromarray(S.image))


@app.route("/api/mask.png")
def api_mask():
    if S.mask is None:
        return _png(Image.new("RGBA", (1, 1), (0, 0, 0, 0)))
    h, w = S.mask.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[S.mask] = (0, 200, 255, 110)
    return _png(Image.fromarray(overlay, "RGBA"))


@app.route("/api/preview.png")
def api_preview():
    """Flattened preview (selected bright, rest dimmed) - handy for Claude to 'see'."""
    if S.image is None:
        abort(404)
    if S.mask is None:
        return _png(Image.fromarray(S.image))
    dim = (S.image * 0.22).astype(np.uint8)
    out = np.where(S.mask[..., None], S.image, dim)
    return _png(Image.fromarray(out))


def resolve_path(p):
    p = (p or "").strip().strip('"').strip("'")
    if not p:
        return p
    if os.path.isabs(p) and os.path.exists(p):
        return p
    for base in (ROOT, os.path.join(ROOT, "images"), os.getcwd()):
        cand = os.path.join(base, p)
        if os.path.exists(cand):
            return cand
    return p


@app.route("/api/load", methods=["POST"])
def api_load():
    data = request.get_json(force=True) or {}
    path = resolve_path(data.get("path", ""))
    if not path or not os.path.exists(path):
        return jsonify({"ok": False, "error": "not found: %s" % path}), 404
    with lock:
        img = Image.open(path).convert("RGB")
        S.image = np.array(img)
        S.size = [img.width, img.height]
        S.image_path = path
        S.points, S.box, S.mask = [], None, None
        SEG.set_image(S.image)
        S.version += 1
    return jsonify({"ok": True, "size": S.size, "path": path})


@app.route("/api/click", methods=["POST"])
def api_click():
    data = request.get_json(force=True)
    with lock:
        if S.image is None:
            return jsonify({"ok": False, "error": "no image"}), 400
        S.points.append([float(data["x"]), float(data["y"]), int(data["label"])])
        recompute()
        S.version += 1
    return jsonify({"ok": True, "points": len(S.points), "has_mask": S.mask is not None})


@app.route("/api/box", methods=["POST"])
def api_box():
    d = request.get_json(force=True)
    box = [min(d["x0"], d["x1"]), min(d["y0"], d["y1"]),
           max(d["x0"], d["x1"]), max(d["y0"], d["y1"])]
    with lock:
        if S.image is None:
            return jsonify({"ok": False, "error": "no image"}), 400
        S.box = [float(v) for v in box]
        recompute()
        S.version += 1
    return jsonify({"ok": True, "has_mask": S.mask is not None})


@app.route("/api/undo", methods=["POST"])
def api_undo():
    with lock:
        if S.points:
            S.points.pop()
        elif S.box is not None:
            S.box = None
        recompute()
        S.version += 1
    return jsonify({"ok": True, "points": len(S.points)})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    with lock:
        S.points, S.box, S.mask = [], None, None
        S.version += 1
    return jsonify({"ok": True})


@app.route("/api/export", methods=["POST"])
def api_export():
    data = request.get_json(force=True) or {}
    with lock:
        if S.image is None or S.mask is None:
            return jsonify({"ok": False, "error": "nothing selected"}), 400
        name = data.get("name")
        if not name:
            stem = os.path.splitext(os.path.basename(S.image_path))[0]
            name = stem + "_cutout.png"
        if not name.lower().endswith(".png"):
            name += ".png"
        rgba = np.dstack([S.image, (S.mask * 255).astype(np.uint8)])
        out_path = os.path.join(OUT_DIR, name)
        Image.fromarray(rgba, "RGBA").save(out_path)
    return jsonify({"ok": True, "path": out_path})


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    print("SegPick on http://127.0.0.1:%d" % args.port)
    app.run(host="127.0.0.1", port=args.port, threaded=True)
