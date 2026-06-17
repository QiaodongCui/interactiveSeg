# CLAUDE.md

Guidance for working in this repo. For user-facing setup/usage see `README.md`.

## What this is

**SegPick** — an interactive tool to cut **one character** out of busy anime/illustration
images. A minimal browser UI (you click to pick the region) plus a **REST API** so an
assistant or script can drive it (load images, add prompts, export). Segmentation is
**HQ-SAM** on the GPU, reusing the model loader from the `isat-sam` package.

## Environment (important — easy to get wrong)

- Windows. **Use the `py` launcher or the venv pythons directly — there is no `python` on PATH.**
- GPU: RTX 4090 (CUDA). HQ-SAM ViT-L uses ~5 GB VRAM.
- **Three local virtualenvs** (all gitignored, not in the repo):

  | venv | purpose | key deps |
  |---|---|---|
  | `.venv` | quick one-shot anime cutouts (CPU) | `rembg` + isnet-anime |
  | `.venv-gpu` | high-res matting | `torch 2.5.1+cu121`, `transformers`, `timm`, `einops`, `kornia` |
  | `.venv-isat` | **SegPick + ISAT SAM backend** | `isat-sam`, `flask`, `torch 2.5.1+cu121` (CUDA) |

- SAM checkpoints live in `.venv-isat/Lib/site-packages/ISAT/checkpoints/`
  (`sam_hq_vit_l.pth`, `sam2.1_hiera_large.pt` are present).

## Common commands

```powershell
# Run SegPick (then open http://127.0.0.1:8765)
.\.venv-isat\Scripts\python.exe segtool\app.py --port 8765

# High-res matting (BiRefNet-matting / HR-matting / ToonOut) on an image
.\.venv-gpu\Scripts\python.exe birefnet_run.py <image> out

# Quick one-shot anime cutout
.\.venv\Scripts\rembg.exe i -m isnet-anime in.png out.png
```

## Architecture

- `segtool/app.py` — Flask backend. Global `State` holds the current image plus a list of
  **objects**, each with its own points/box/mask/logits (independent SAM predictions, no
  cross-talk) and an active-object pointer. One `SegAny` (HQ-SAM) instance; GPU calls are
  serialized with a lock. Each object refines **incrementally** (prior low-res logits fed
  back as `mask_input`). REST API: `POST /api/{load,upload,click,box,undo,reset,granularity,
  export,object/{new,select,delete,rename}}`, `GET /api/{state,image.png,mask.png,preview.png,
  cutout.png}`. `/api/preview.png` = union of masks bright / rest dimmed (for programmatic
  verification); `/api/cutout.png?mode=merged|active&obj=<id>` returns cutout bytes for the
  browser file-save dialog.
- `segtool/static/index.html` — single-page UI. Left-click=add, right-click=remove, drag=box,
  wheel=zoom, middle-drag=pan; object chips to add/select/delete/rename; `g` cycles
  granularity; exports use the native file picker. Polls `/api/state` every 700 ms so
  external (API-driven) changes show up live.
- `birefnet_run.py` — BiRefNet matting + ToonOut anime fine-tune via HuggingFace `transformers`.
- `isat_smoketest.py`, `isat_refine.py`, `diag_toonout.py` — dev/experiment scripts.

## Gotchas

- **ISAT torch must be CUDA but stable: `2.5.1+cu121`.** The newer `cu128` build caused
  intermittent `WinError 1114` (`c10.dll` init failure) when launching the GUI/server.
  ISAT itself only requires `torch>=2.1.1`.
- **ToonOut weights** (`joelseytre/toonout`, `birefnet_finetuned_toonout.pth`) have a
  `module._orig_mod.` prefix (DDP + `torch.compile`) that must be stripped before loading
  into the HF `ZhengPeng7/BiRefNet` architecture — handled in `birefnet_run.py`.
- SegPick export is a **binary** SAM mask. For soft/wispy hair, pipe the cutout through
  `birefnet_run.py`.
- **One-shot anime models cannot isolate one character** from busy/decorated multi-character
  images — that's what SegPick's interactive box+click flow is for.
- **HQ-SAM collapses multimask granularities.** `predict(multimask_output=True)` returns a
  single mask (the HQ token's logits veto the larger SAM scopes), so the subpart/part/whole
  toggle can't use the public API. `_granularity_candidates()` calls the HQ
  `mask_decoder.predict_masks` directly to recover the 3 raw SAM scopes; `auto` (default)
  keeps the HQ-quality mask. Wrapped in try/except → falls back to the single mask.

## Conventions

- `out/` (generated cutouts) and `images/` (inputs) are gitignored.
- **Do not `git commit` or `git push` without explicit confirmation.**
- Match the style of surrounding code; keep the tool dependency-light.
