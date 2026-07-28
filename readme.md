# Zero-Shot Auto-Labelling Pipeline

Config-driven pipeline that runs one or more zero-shot / open-vocabulary detectors over a
folder of images and exports a YOLO-format dataset (images, labels, `data.yaml`, optional
annotated previews) ready to drop straight into training.

Supports six labellers you can mix and match in a single run, then merges + NMS-dedupes
their outputs into one dataset:

| Model | Type | Notes |
|---|---|---|
| `DINO` | GroundingDINO (SwinT/SwinB) | phrase grounding, multi-scale inference + NMS |
| `FLORANCE` | Florence-2-large | open-vocab detection + caption-to-phrase grounding |
| `YOLO` | your own trained `.pt` | needs `yolo_model_path` |
| `YOLOWORLD` | YOLO-World | open-vocab YOLO, no training needed |
| `OWL2` | OWLv2 | open-vocab, good general baseline |
| `LOCATEANYTHING` | NVIDIA LocateAnything-3B | generative VLM grounding, no confidence scores |

---

## Setup

```bash
pip install torch torchvision ultralytics transformers opencv-python-headless \
            tqdm pyyaml pillow
```

That covers the base requirements. Each model in the table above has its own extra
dependencies, weight downloads, and (for `DINO`/`FLORANCE`) a local checkout the pipeline
expects a path to — full step-by-step instructions per model are in
**[MODELS.md](./MODELS.md)**. Only install what you actually list in `models2run`.

---

## Folder layout

```
<script folder>/
├── nvidia-zeroshot.py
├── Zeroshot-Images4Labelling/   ← input: put batches here
│   ├── BATCH1/*.jpg
│   └── BATCH2/*.jpg
└── Zeroshot-LabelledOutput/     ← output: created automatically
    └── dataset_0_2026-07-28/
        ├── raw/inference.json        # merged COCO-style detections
        ├── images/                   # copied source frames
        ├── labels/                   # YOLO .txt labels
        ├── annotated/                # preview images w/ drawn boxes (if enabled)
        └── data.yaml                 # YOLO training config
```

---

## Running

```bash
python nvidia-zeroshot.py --config my_config.json
```

With no `--config`, it falls back to the defaults hardcoded near the top of the script
(edit the `_cfg` dict directly for quick one-off runs).

---

## Config reference

Only `items` is required — everything else has a default.

| Key | Type | Default | Description |
|---|---|---|---|
| `items` | list/dict | — | **required.** See "Class configuration" below |
| `yolo_model_path` | str | `""` | path to your `.pt`, required if `"YOLO"` in `models2run` |
| `models2run` | list[str] | `["YOLO"]` | any of `DINO`, `FLORANCE`, `YOLO`, `OWL2`, `YOLOWORLD`, `LOCATEANYTHING` |
| `batch_selection` | list[str] | `["all"]` | batch folder names, or `["all"]` |
| `box_threshold` | float | `0.45` | DINO/YOLO/YOLO-World box confidence cutoff |
| `text_threshold` | float | `0.15` | DINO text-match cutoff |
| `merge_nms_iou` | float | `0.4` | IoU threshold when de-duping across models |
| `florence_score_threshold` | float | `0.3` | Florence-2 detection score cutoff |
| `owl_score_threshold` | float | `0.1` | OWLv2 detection score cutoff |
| `dino_type` | str | `"SwinB"` | `"SwinT"` or `"SwinB"` |
| `annotated` | bool | `true` | save preview images with drawn boxes |
| `assessment` | bool | `false` | print per-model frame/box breakdown after the run |
| `grounding_dino_dir` | str | `<parent>/GroundingDINO` | override if your checkout lives elsewhere |
| `florence_path` | str | `<parent>/Florence-2-large` | override if your checkout lives elsewhere |
| `yolo_world_model` | str | `"yolov8x-worldv2.pt"` | YOLO-World checkpoint |
| `owl_model` | str | `"google/owlv2-large-patch14-ensemble"` | OWLv2 checkpoint |
| `locateanything_model_path` | str | `"nvidia/LocateAnything-3B"` | HF repo id or local path |
| `locateanything_generation_mode` | str | `"hybrid"` | `"fast"` / `"slow"` / `"hybrid"` |
| `locateanything_max_new_tokens` | int | `2048` | generation length cap |
| `locateanything_max_image_side` | int | `1024` | downscale images to this before inference (prevents CUDA OOM); `0` disables |
| `locateanything_do_sample` | bool | `true` | `false` = greedy decoding, deterministic, usually more conservative |
| `locateanything_temperature` | float | `0.7` | sampling temperature (only used if `do_sample: true`) |
| `locateanything_top_p` | float | `0.9` | nucleus sampling cutoff |
| `locateanything_repetition_penalty` | float | `1.1` | discourages repeated box spam |

> **Note on LocateAnything "confidence":** it's a generative VLM, not a scored detector — it
> writes `<box>` tokens as text rather than assigning each box a confidence value. There's no
> `locateanything_score_threshold` because that number doesn't exist upstream. Use
> `locateanything_do_sample: false` for tighter, more repeatable boxes instead.

---

## Class configuration (`items`)

Three accepted formats, all mixable across a project as you scale up:

**1. Flat list** — every prompt is its own class:
```json
"items": ["tanks", "trucks", "cans", "soda"]
```

**2. Grouped by dict** — multiple prompts collapse into one output class:
```json
"items": {
  "vehicle":   ["tanks", "trucks", "vehicles", "cars"],
  "container": ["cans", "soda"]
}
```

**3. Grouped by list-of-objects** — same as (2), alternate spelling:
```json
"items": [
  {"label": "vehicle",   "prompts": ["tanks", "trucks", "vehicles", "cars"]},
  {"label": "container", "prompts": ["cans", "soda"]}
]
```

Every model still gets queried with each individual prompt (so detection recall isn't
reduced), but all matching detections are written under the merged class in the final
`data.yaml` / label files. `category_id`s are always assigned 0-indexed in the order classes
appear — dict/list order is preserved, but a literal id like `"1"` used as a dict key is just
a display label, not the numeric id written to the `.txt` files.

The console prints a class map at the start of every run so you can sanity-check the
grouping before inference kicks off:
```
── Classes ──────────────────────────────────────────
  [0] vehicle   <-  ['tanks', 'trucks', 'vehicles', 'cars']
  [1] container <-  ['cans', 'soda']
──────────────────────────────────────────────────────
```

---

## Output

- `raw/inference.json` — merged COCO-style detections (`meta`, `categories`, `images`,
  `annotations`) with confidence + source model per box.
- `images/` + `labels/` — standard YOLO training pair, one `.txt` per image
  (`class_id cx cy w h`, normalized).
- `annotated/` — same images with boxes + labels drawn on, if `annotated: true`.
- `data.yaml` — ready to point `ultralytics` training at directly.
- If `assessment: true`, a per-model frame/box breakdown prints after export, plus the final
  merged+NMS counts.
- After export, you're prompted to manually walk through and correct labels image-by-image
  in the terminal (`y`/`N`).
- Datasets that end up with an empty `annotated/` folder are auto-deleted at exit (nothing
  detected == nothing worth keeping).

---

## Troubleshooting

**`FileNotFoundError: .../GroundingDINO/groundingdino/config/...`**
Your GroundingDINO checkout isn't where the script guessed. Set `grounding_dino_dir`
explicitly in your config to the actual path.

**`torch.OutOfMemoryError` on LocateAnything**
Almost always a large source image blowing up ViT attention memory. Lower
`locateanything_max_image_side` (try `768` or `512`). The script also sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and clears CUDA cache between frames
automatically; OOM on a single image/prompt now skips and continues rather than killing the
whole run.

**Nothing detected / empty dataset**
Check `box_threshold` / `*_score_threshold` aren't too strict, and that your `items` prompts
actually match how the object appears (e.g. `"standing soda can"` vs just `"can"`).