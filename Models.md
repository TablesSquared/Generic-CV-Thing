# Model Installation Guide

Per-model setup for every labeller supported by the zero-shot pipeline. You only need to
install the ones you list in `models2run` — everything is imported lazily inside each
`Labeller*` function, so unused models never need their dependencies present.

Base requirements (needed regardless of which models you run):

```bash
pip install torch torchvision ultralytics transformers opencv-python-headless \
            tqdm pyyaml pillow
```

Use a CUDA build of `torch`/`torchvision` matching your driver — check with `nvidia-smi`
first.

---

## GroundingDINO (`DINO`)

Not on PyPI as a normal package — you build it from source, then point the pipeline's
`grounding_dino_dir` config key at the checkout.

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
```

Download whichever backbone(s) you plan to use into a `weights/` folder inside the checkout:

```bash
mkdir -p weights && cd weights

# SwinT (smaller, faster)
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# SwinB (larger, more accurate — used by default in the pipeline config)
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth

cd ..
```

Then in your pipeline config:
```json
"grounding_dino_dir": "/absolute/path/to/GroundingDINO",
"dino_type": "SwinB"
```

The pipeline expects the configs at
`GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py` and
`GroundingDINO_SwinB_cfg.py`, and the weights at
`GroundingDINO/weights/groundingdino_swint_ogc.pth` /
`groundingdino_swinb_cogcoor.pth` — both come with the repo clone / the download above, no
renaming needed.

---

## Florence-2-large (`FLORANCE`)

Pulled through `transformers`, but the pipeline expects a **local** checkout rather than
downloading it fresh every run:

```bash
pip install einops timm
python -c "
from transformers import AutoProcessor, AutoModelForCausalLM
AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True).save_pretrained('./Florence-2-large')
AutoProcessor.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True).save_pretrained('./Florence-2-large')
"
```

Then in your config:
```json
"florence_path": "/absolute/path/to/Florence-2-large"
```

---

## YOLO (`YOLO`)

Your own trained checkpoint — no separate install beyond `ultralytics` (already in the base
requirements). Just point at the weights:

```json
"yolo_model_path": "/absolute/path/to/best.pt"
```

---

## YOLO-World (`YOLOWORLD`)

Ships as part of `ultralytics`, so nothing extra to install. The checkpoint auto-downloads
on first use:

```json
"yolo_world_model": "yolov8x-worldv2.pt"
```

If you'd rather pre-download it:
```bash
python -c "from ultralytics import YOLOWorld; YOLOWorld('yolov8x-worldv2.pt')"
```

---

## OWLv2 (`OWL2`)

Pure `transformers`, auto-downloads from the Hugging Face Hub on first run — nothing extra
to install:

```json
"owl_model": "google/owlv2-large-patch14-ensemble"
```

Pre-download if you want to avoid a cold start mid-run:
```bash
python -c "
from transformers import Owlv2Processor, Owlv2ForObjectDetection
Owlv2Processor.from_pretrained('google/owlv2-large-patch14-ensemble')
Owlv2ForObjectDetection.from_pretrained('google/owlv2-large-patch14-ensemble')
"
```

---

## NVIDIA LocateAnything-3B (`LOCATEANYTHING`)

Needs CUDA 12.2+ and a fairly specific dependency set — version-pin these, the
`trust_remote_code` model code is picky:

```bash
pip install transformers==4.57.1 numpy==1.26.4 Pillow==11.1.0 \
            opencv-python-headless==4.11.0.86 torchvision peft \
            decord==0.6.0 lmdb==1.7.5
```

Weights (~7.7 GB) auto-download from Hugging Face on first run and cache locally:

```json
"locateanything_model_path": "nvidia/LocateAnything-3B"
```

Pre-download if you want to avoid the wait mid-run:
```bash
python -c "
from transformers import AutoModel, AutoTokenizer, AutoProcessor
for cls in (AutoModel, AutoTokenizer, AutoProcessor):
    cls.from_pretrained('nvidia/LocateAnything-3B', trust_remote_code=True)
"
```

Needs ~8 GB VRAM in bf16. If you're on a smaller card, see the
`locateanything_max_image_side` / OOM guidance in the main [README](./README.md).

License is **non-commercial research use only** (NVIDIA Open Model License) — fine for
internal labelling/experimentation, not for a shipped product.

---

## Verifying an install

Quick sanity check after installing any model — run with just that one model selected and a
tiny batch (1-2 images) in `models2run` before doing a full run, so you catch missing
weights/paths early instead of 20 minutes into a big batch.