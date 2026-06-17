---
name: segpick
description: Drive the SegPick interactive segmentation tool via its REST API — start the server, load an image, add point/box prompts, view the current selection, and export a transparent cutout. Use when the user asks Claude to load an image into the tool, operate/seed the segmentation, check what's selected, or export a result.
---

# Driving SegPick

SegPick is a local Flask app (`segtool/app.py`) wrapping HQ-SAM. The user picks regions
in the browser at <http://127.0.0.1:8765>; Claude drives the same state over HTTP. Default
port `8765`. Coordinates are **original image pixels** (get size from `/api/state`).

## 1. Make sure the server is up

```powershell
try { Invoke-RestMethod http://127.0.0.1:8765/api/state -TimeoutSec 4 | ConvertTo-Json -Compress } catch { "DOWN" }
```

If down, start it (background) and poll until `/api/state` responds:

```powershell
.\.venv-isat\Scripts\python.exe segtool\app.py --port 8765   # run_in_background: true
```

## 2. Drive it

```powershell
$b = "http://127.0.0.1:8765"
function Post($p,$body){ Invoke-RestMethod -Uri "$b$p" -Method Post -ContentType 'application/json' -Body $body }

Post "/api/load"  '{"path":"images/Sandrone_Card.png"}'   # relative to project root or images/, or absolute
Post "/api/box"   '{"x0":80,"y0":110,"x1":710,"y1":1780}' # box prompt
Post "/api/click" '{"x":210,"y":600,"label":1}'           # label 1 = add, 0 = remove
Post "/api/undo"  '{}'
Post "/api/reset" '{}'
Post "/api/export" '{}'                                    # -> out\<stem>_cutout.png  (optional {"name":"foo.png"})
Invoke-RestMethod "$b/api/state" | ConvertTo-Json -Compress
```

## 3. "See" the current selection

Fetch the flattened preview (selected bright / rest dimmed) and Read it:

```powershell
Invoke-WebRequest "$b/api/preview.png" -OutFile out\_preview.png
```

Then Read `out\_preview.png` to verify before exporting.

## Notes
- The browser polls every 700 ms, so anything Claude changes shows up live for the user.
- Don't `reset` or overwrite the user's in-progress selection unless asked.
- Export is a binary mask. For soft hair: `.\.venv-gpu\Scripts\python.exe birefnet_run.py out\<cutout>.png out`.
- Typical division of labor: Claude loads images / seeds a box; the user refines by clicking; export when done.
