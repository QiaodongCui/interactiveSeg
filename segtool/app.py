"""
SegPick - minimal interactive character-segmentation tool (multi-object).

Backend: reuses ISAT's SegAny (HQ-SAM on GPU). The image embedding is computed
once per image; each *object* keeps its own points/box/mask/logits and gets an
independent SAM prediction (so objects never interfere with each other).
Frontend: minimal web UI (static/index.html). Control: REST API for human + Claude.

Run:  .venv-isat\Scripts\python.exe segtool\app.py --port 8765
Then open http://127.0.0.1:8765
"""
import os
import io
import re
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

PALETTE = [
    (0, 200, 255), (255, 140, 0), (120, 255, 0), (255, 0, 160),
    (170, 120, 255), (255, 210, 0), (0, 255, 180), (255, 90, 90),
]

app = Flask(__name__, static_folder=os.path.join(HERE, "static"), static_url_path="")
lock = threading.Lock()


class Obj:
    _seq = 0

    def __init__(self):
        Obj._seq += 1
        self.id = Obj._seq
        self.name = "object %d" % self.id
        self.color = PALETTE[(self.id - 1) % len(PALETTE)]
        self.points = []     # [[x, y, label], ...]
        self.box = None      # [x0, y0, x1, y1]
        self.mask = None     # np bool HxW
        self.logits = None   # low-res logits for incremental refinement

    def summary(self):
        return {
            "id": self.id, "name": self.name, "color": list(self.color),
            "npoints": len(self.points), "has_box": self.box is not None,
            "coverage": round(100.0 * float(self.mask.mean()), 2) if self.mask is not None else 0,
        }


class State:
    def __init__(self):
        self.version = 0
        self.image_path = None
        self.image = None
        self.size = [0, 0]
        self.objects = []
        self.active = -1

    def new_image(self, image, path):
        Obj._seq = 0
        self.image = image
        self.size = [image.shape[1], image.shape[0]]
        self.image_path = path
        self.objects = [Obj()]
        self.active = 0

    def cur(self):
        return self.objects[self.active] if 0 <= self.active < len(self.objects) else None


S = State()

print("Loading SAM:", DEFAULT_CKPT)
SEG = SegAny(DEFAULT_CKPT, use_bfloat16=True)
print("SAM ready on", SEG.device)


def recompute(obj, incremental=True):
    """Run SAM with one object's prompts -> obj.mask (independent of other objects)."""
    if S.image is None or obj is None or (not obj.points and obj.box is None):
        if obj is not None:
            obj.mask = None
            obj.logits = None
        return
    pts = np.array([[p[0], p[1]] for p in obj.points], dtype=float) if obj.points else None
    lbls = np.array([p[2] for p in obj.points], dtype=int) if obj.points else None
    box = np.array(obj.box, dtype=float) if obj.box is not None else None
    mask_input = obj.logits[None, :, :] if (incremental and obj.logits is not None) else None
    multimask = mask_input is None and box is None and len(obj.points) == 1
    with torch.inference_mode(), torch.autocast(
        SEG.device, dtype=SEG.model_dtype, enabled=torch.cuda.is_available()
    ):
        masks, scores, logits = SEG.predictor.predict(
            point_coords=pts, point_labels=lbls, box=box,
            mask_input=mask_input, multimask_output=multimask,
        )
    masks = np.asarray(masks)
    logits = np.asarray(logits)
    best = int(np.argmax(scores)) if multimask else 0
    obj.mask = (masks[best] if masks.ndim == 3 else masks).astype(bool)
    obj.logits = logits[best].astype(np.float32)


def bump():
    S.version += 1


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/state")
def api_state():
    cur = S.cur()
    return jsonify({
        "version": S.version,
        "image_path": S.image_path,
        "size": S.size,
        "model": os.path.basename(DEFAULT_CKPT),
        "device": SEG.device,
        "active": S.active,
        "objects": [o.summary() for o in S.objects],
        "points": cur.points if cur else [],
        "box": cur.box if cur else None,
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
    """All objects, each in its own color; the active object drawn on top + brighter."""
    if S.image is None:
        return _png(Image.new("RGBA", (1, 1), (0, 0, 0, 0)))
    h, w = S.image.shape[:2]
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    order = [i for i in range(len(S.objects)) if i != S.active] + ([S.active] if S.cur() else [])
    for i in order:
        o = S.objects[i]
        if o.mask is None:
            continue
        a = 150 if i == S.active else 90
        overlay[o.mask] = (o.color[0], o.color[1], o.color[2], a)
    return _png(Image.fromarray(overlay, "RGBA"))


@app.route("/api/preview.png")
def api_preview():
    """Union of all object masks bright, rest dimmed (for Claude to 'see')."""
    if S.image is None:
        abort(404)
    union = None
    for o in S.objects:
        if o.mask is not None:
            union = o.mask if union is None else (union | o.mask)
    if union is None:
        return _png(Image.fromarray(S.image))
    dim = (S.image * 0.22).astype(np.uint8)
    out = np.where(union[..., None], S.image, dim)
    return _png(Image.fromarray(out))


@app.route("/api/cutout.png")
def api_cutout():
    """RGBA cutout bytes for browser download. ?mode=merged|active or ?obj=<id>."""
    if S.image is None:
        abort(404)
    obj_id = request.args.get("obj")
    mode = request.args.get("mode", "merged")
    if obj_id is not None:
        o = next((x for x in S.objects if str(x.id) == obj_id), None)
        mask = o.mask if o else None
    elif mode == "active":
        cur = S.cur()
        mask = cur.mask if cur else None
    else:
        mask = None
        for o in S.objects:
            if o.mask is not None:
                mask = o.mask.copy() if mask is None else (mask | o.mask)
    if mask is None:
        abort(404)
    rgba = np.dstack([S.image, (mask * 255).astype(np.uint8)])
    return _png(Image.fromarray(rgba, "RGBA"))


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


def _set_image(image, path):
    with lock:
        S.new_image(image, path)
        SEG.set_image(S.image)
        bump()


@app.route("/api/load", methods=["POST"])
def api_load():
    data = request.get_json(force=True) or {}
    path = resolve_path(data.get("path", ""))
    if not path or not os.path.exists(path):
        return jsonify({"ok": False, "error": "not found: %s" % path}), 404
    _set_image(np.array(Image.open(path).convert("RGB")), path)
    return jsonify({"ok": True, "size": S.size, "path": path})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify({"ok": False, "error": "no file"}), 400
    _set_image(np.array(Image.open(f.stream).convert("RGB")), f.filename)
    return jsonify({"ok": True, "size": S.size, "path": f.filename})


@app.route("/api/click", methods=["POST"])
def api_click():
    data = request.get_json(force=True)
    with lock:
        cur = S.cur()
        if S.image is None or cur is None:
            return jsonify({"ok": False, "error": "no image"}), 400
        cur.points.append([float(data["x"]), float(data["y"]), int(data["label"])])
        recompute(cur, incremental=True)
        bump()
    return jsonify({"ok": True, "points": len(cur.points)})


@app.route("/api/box", methods=["POST"])
def api_box():
    d = request.get_json(force=True)
    box = [min(d["x0"], d["x1"]), min(d["y0"], d["y1"]),
           max(d["x0"], d["x1"]), max(d["y0"], d["y1"])]
    with lock:
        cur = S.cur()
        if S.image is None or cur is None:
            return jsonify({"ok": False, "error": "no image"}), 400
        cur.box = [float(v) for v in box]
        cur.logits = None  # a new box is a fresh anchor
        recompute(cur, incremental=True)
        bump()
    return jsonify({"ok": True})


@app.route("/api/undo", methods=["POST"])
def api_undo():
    with lock:
        cur = S.cur()
        if cur is None:
            return jsonify({"ok": False, "error": "no object"}), 400
        if cur.points:
            cur.points.pop()
        elif cur.box is not None:
            cur.box = None
        cur.logits = None
        recompute(cur, incremental=False)
        bump()
    return jsonify({"ok": True, "points": len(cur.points)})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Clear the active object's prompts (keeps the object slot)."""
    with lock:
        cur = S.cur()
        if cur is not None:
            cur.points, cur.box, cur.mask, cur.logits = [], None, None, None
        bump()
    return jsonify({"ok": True})


@app.route("/api/object/new", methods=["POST"])
def api_object_new():
    with lock:
        if S.image is None:
            return jsonify({"ok": False, "error": "no image"}), 400
        S.objects.append(Obj())
        S.active = len(S.objects) - 1
        bump()
    return jsonify({"ok": True, "active": S.active})


@app.route("/api/object/select", methods=["POST"])
def api_object_select():
    data = request.get_json(force=True) or {}
    i = int(data.get("index", -1))
    with lock:
        if not (0 <= i < len(S.objects)):
            return jsonify({"ok": False, "error": "bad index"}), 400
        S.active = i
        bump()
    return jsonify({"ok": True, "active": S.active})


@app.route("/api/object/delete", methods=["POST"])
def api_object_delete():
    data = request.get_json(force=True) or {}
    i = int(data.get("index", S.active))
    with lock:
        if not (0 <= i < len(S.objects)):
            return jsonify({"ok": False, "error": "bad index"}), 400
        S.objects.pop(i)
        if not S.objects:
            S.objects = [Obj()]
        S.active = min(S.active, len(S.objects) - 1)
        bump()
    return jsonify({"ok": True, "active": S.active})


@app.route("/api/object/rename", methods=["POST"])
def api_object_rename():
    data = request.get_json(force=True) or {}
    i = int(data.get("index", S.active))
    name = str(data.get("name", "")).strip()
    with lock:
        if not (0 <= i < len(S.objects)) or not name:
            return jsonify({"ok": False, "error": "bad request"}), 400
        S.objects[i].name = name
        bump()
    return jsonify({"ok": True})


def _safe(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "object"


@app.route("/api/export", methods=["POST"])
def api_export():
    """mode: 'merged' (union of all objects, default) | 'active' | 'each'."""
    data = request.get_json(force=True) or {}
    mode = data.get("mode", "merged")
    with lock:
        if S.image is None:
            return jsonify({"ok": False, "error": "no image"}), 400
        stem = os.path.splitext(os.path.basename(S.image_path))[0]
        masks = [(o, o.mask) for o in S.objects if o.mask is not None]
        if not masks:
            return jsonify({"ok": False, "error": "nothing selected"}), 400

        def save(mask, name):
            rgba = np.dstack([S.image, (mask * 255).astype(np.uint8)])
            path = os.path.join(OUT_DIR, name)
            Image.fromarray(rgba, "RGBA").save(path)
            return path

        if mode == "each":
            paths = [save(m, "%s__%s.png" % (stem, _safe(o.name))) for o, m in masks]
            return jsonify({"ok": True, "paths": paths})
        if mode == "active":
            cur = S.cur()
            if cur is None or cur.mask is None:
                return jsonify({"ok": False, "error": "active object empty"}), 400
            return jsonify({"ok": True, "paths": [save(cur.mask, "%s__%s.png" % (stem, _safe(cur.name)))]})
        # merged
        union = masks[0][1].copy()
        for _, m in masks[1:]:
            union |= m
        return jsonify({"ok": True, "paths": [save(union, "%s_cutout.png" % stem)]})


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    print("SegPick on http://127.0.0.1:%d" % args.port)
    app.run(host="127.0.0.1", port=args.port, threaded=True)
