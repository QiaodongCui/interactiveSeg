# interactiveSeg — SegPick

A minimal, interactive tool for cutting **one character** out of busy anime/illustration
images (e.g. game promo art with non-trivial hair and clothing boundaries).

It pairs a clean browser UI with a **REST API**, so a human picks the region by
clicking while an assistant (or any script) can drive it programmatically — load
images, add prompts, export cutouts.

Backed by **HQ-SAM** (Segment Anything in High Quality) running on an NVIDIA GPU,
reusing the model loader from [ISAT](https://github.com/yatengLG/ISAT_with_segment_anything).

```
You (browser):  left-click = add · right-click = remove · drag = box  → live mask
Assistant/API:  POST /api/load · /api/click · /api/box · /api/export   → automation
```

## Why

For *designed* images (decorative backgrounds, multiple characters), one-shot
background removers can't know **which** character you want, and over- or
under-segment. Interactive prompting (box + a few clicks) reliably isolates a single
character. See [Pipeline](#pipeline-for-best-quality) for fine-hair refinement.

## Requirements

- Windows (tested) / Linux, Python 3.12
- NVIDIA GPU (developed on an RTX 4090; HQ-SAM ViT-L uses ~5 GB VRAM)

## Setup

```bash
python -m venv .venv-isat
.venv-isat\Scripts\activate            # Windows  (source .venv-isat/bin/activate on Linux)

# 1) Install CUDA torch FIRST (pick the index for your CUDA; cu121 is a stable choice)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# 2) Install the rest
pip install -r requirements.txt
```

Download a SAM checkpoint into `ISAT/checkpoints/` inside the venv (HQ-SAM ViT-L
recommended). The easiest way is to run ISAT's model manager once (`isat-sam`),
or fetch it directly:

```
https://huggingface.co/yatengLG/ISAT_with_segment_anything_checkpoints/resolve/main/sam_hq_vit_l.pth
-> .venv-isat/Lib/site-packages/ISAT/checkpoints/sam_hq_vit_l.pth
```

## Run

```bash
.venv-isat\Scripts\python.exe segtool\app.py --port 8765
```

Open <http://127.0.0.1:8765>. Put your own images anywhere and load them by path.

### UI

| Action | Result |
|---|---|
| Left-click | add region (positive point) |
| Right-click | remove region (negative point) |
| Drag | box prompt |
| `z` / `r` / `e` | undo / reset / export |

Exports are transparent PNG cutouts written to `out/`.

### REST API

| Method | Endpoint | Body |
|---|---|---|
| POST | `/api/load` | `{ "path": "images/foo.png" }` |
| POST | `/api/click` | `{ "x":123, "y":45, "label":1 }` (1=add, 0=remove) |
| POST | `/api/box` | `{ "x0":..,"y0":..,"x1":..,"y1":.. }` |
| POST | `/api/undo` / `/api/reset` | — |
| POST | `/api/export` | `{ "name": "foo_cutout.png" }` (optional) |
| GET | `/api/state` | current prompts, size, mask coverage |
| GET | `/api/image.png` / `/api/mask.png` / `/api/preview.png` | image / overlay / flattened preview |

Example (PowerShell):

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8765/api/load -Method Post `
  -ContentType 'application/json' -Body '{"path":"images/character.png"}'
```

## Pipeline (for best quality)

The SegPick export is a crisp **binary** SAM mask. For soft, wispy hair, refine the
crop with high-resolution matting (BiRefNet matting checkpoints / the ToonOut anime
fine-tune). `birefnet_run.py` shows this; it needs a separate env with
`transformers timm einops kornia safetensors huggingface_hub` plus CUDA torch.

```
ISAT/SegPick (box + clicks)  →  isolate & crop the character  →  BiRefNet-matting  →  soft-alpha PNG
```

## Files

| Path | What |
|---|---|
| `segtool/app.py` | Flask backend + SAM + REST API |
| `segtool/static/index.html` | minimal single-page UI |
| `birefnet_run.py` | BiRefNet-matting / ToonOut high-res matting (separate env) |
| `isat_smoketest.py`, `isat_refine.py`, `diag_toonout.py` | dev/experiment scripts |

## Credits

- [Segment Anything in High Quality (HQ-SAM)](https://github.com/SysCV/sam-hq)
- [ISAT_with_segment_anything](https://github.com/yatengLG/ISAT_with_segment_anything) — SAM backend / checkpoints
- [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) · [ToonOut](https://github.com/MatteoKartoon/BiRefNet)
