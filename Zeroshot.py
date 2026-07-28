import sys
import warnings
warnings.filterwarnings("ignore")
import os
import argparse
import torch
import cv2
from tqdm import tqdm
import yaml
from datetime import datetime
import json
import shutil
from PIL import Image

BaseDir     = os.path.dirname(os.path.abspath(__file__))            # this script's folder
BaseBaseDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # parent project folder

# ── GPU setup ──────────────────────────────────────────────────────────────────
os.environ["CUDA_DEVICE_ORDER"]      = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]   = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.backends.cudnn.benchmark        = True
torch.backends.cudnn.deterministic    = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

if torch.cuda.is_available():
    _device = torch.device("cuda")
    _ = torch.zeros(1, device=_device)
    torch.cuda.empty_cache()
    print(f"GPU : {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"CUDA: {torch.version.cuda}")
else:
    print("No GPU found, running on CPU")

# ── CLI / config ──────────────────────────────────────────────────────────────
# Input images go in:  ./zeroshot-images4labelling/<batch_name>/*.jpg
# Output datasets go in: ./zeroshot-labelledoutputs/dataset_N_<date>/
#
# Config keys (all optional except 'items'):
#   items                 list[str]  - classes/prompts to detect (REQUIRED)
#   yolo_model_path       str        - path to a .pt YOLO model, used if "YOLO" in models2run
#   models2run            list[str]  - subset of ["DINO","FLORANCE","YOLO","OWL2","YOLOWORLD"]
#   batch_selection       list[str]  - batch folder names to process, or ["all"]
#   box_threshold         float
#   text_threshold        float
#   merge_nms_iou         float
#   florence_score_threshold float
#   owl_score_threshold      float
#   dino_type             str        - "SwinT" or "SwinB"
#   annotated             bool       - save preview images with drawn boxes
parser = argparse.ArgumentParser(description="Generalized zero-shot auto-labelling pipeline")
parser.add_argument("--config", default=None, help="Path to a JSON config file")
args = parser.parse_args()


if args.config and os.path.isfile(args.config):
    with open(args.config, "r") as f:
        _cfg = json.load(f)
    print(f"[INFO]  Loaded config from {args.config}")

# ██████╗░██████╗░░█████╗░░██████╗░██████╗░░█████╗░███╗░░░███╗  ░█████╗░░█████╗░███╗░░██╗███████╗██╗░██████╗░
# ██╔══██╗██╔══██╗██╔══██╗██╔════╝░██╔══██╗██╔══██╗████╗░████║  ██╔══██╗██╔══██╗████╗░██║██╔════╝██║██╔════╝░
# ██████╔╝██████╔╝██║░░██║██║░░██╗░██████╔╝███████║██╔████╔██║  ██║░░╚═╝██║░░██║██╔██╗██║█████╗░░██║██║░░██╗░
# ██╔═══╝░██╔══██╗██║░░██║██║░░╚██╗██╔══██╗██╔══██║██║╚██╔╝██║  ██║░░██╗██║░░██║██║╚████║██╔══╝░░██║██║░░╚██╗
# ██║░░░░░██║░░██║╚█████╔╝╚██████╔╝██║░░██║██║░░██║██║░╚═╝░██║  ╚█████╔╝╚█████╔╝██║░╚███║██║░░░░░██║╚██████╔╝
# ╚═╝░░░░░╚═╝░░╚═╝░╚════╝░░╚═════╝░╚═╝░░╚═╝╚═╝░░╚═╝╚═╝░░░░░╚═╝  ░╚════╝░░╚════╝░╚═╝░░╚══╝╚═╝░░░░░╚═╝░╚═════╝░
# edit here to change zeroshot config
else:
    print("[WARN]  No --config supplied; using script-level defaults below.")
    _cfg = {
        # ── Edit these for a direct/standalone run ───────────────────────────
        "items":           ["standing soda can"],
        "yolo_model_path": "/home/tables/Documents/TitaThing/TitaSharedHost/TitaModels/model_35_2026-06-03/train/weights/best.pt",
        "models2run":      ["YOLO"],
        "batch_selection": ["all"],
        "box_threshold":   0.45,
        "text_threshold":  0.15,
        "merge_nms_iou":   0.4,
        "florence_score_threshold": 0.3,
        "owl_score_threshold":      0.2,
        "dino_type":       "SwinB",
        "annotated":       True,
    }

items = _cfg.get("items")
if not items:
    raise ValueError(
        "No 'items' (class/prompt list) provided. Pass via --config "
        "with an 'items' key, e.g. {\"items\": [\"standing soda can\"]}"
    )

CAMERA_FEED_PATH = os.path.join(BaseDir, "zeroshot-images4labelling")
OUTPUT_PATH      = os.path.join(BaseDir, "zeroshot-labelledoutputs")
YOLO_MODEL_PATH  = _cfg.get("yolo_model_path", "")

BOX_TRESHOLD  = float(_cfg.get("box_threshold", 0.45))
TEXT_TRESHOLD = float(_cfg.get("text_threshold", 0.15))
MERGE_NMS_IOU = float(_cfg.get("merge_nms_iou", 0.4))

OWL_BATCH_SIZE      = int(_cfg.get("owl_batch_size", 4))
FLORENCE_BATCH_SIZE = int(_cfg.get("florence_batch_size", 4))
DINO_BATCH_SIZE     = int(_cfg.get("dino_batch_size", 1))
YOLO_BATCH_SIZE     = int(_cfg.get("yolo_batch_size", 8))

BATCH_SELECTION = _cfg.get("batch_selection", ["all"])

annotated = bool(_cfg.get("annotated", True))

FLORENCE_SCORE_THRESHOLD = float(_cfg.get("florence_score_threshold", 0.3))
OWL_SCORE_THRESHOLD      = float(_cfg.get("owl_score_threshold", 0.2))

DINO_TYPE = _cfg.get("dino_type", "SwinB")

models2run = _cfg.get("models2run", ["YOLO"])

# Paths to external model assets — only needed if the corresponding model is selected
GROUNDING_DINO_DIR = _cfg.get("grounding_dino_dir", os.path.join(BaseBaseDir, "GroundingDINO"))
FLORENCE_PATH       = _cfg.get("florence_path", os.path.join(BaseBaseDir, "Florence-2-large"))
YOLO_WORLD_MODEL    = _cfg.get("yolo_world_model", "yolov8x-worldv2.pt")
OWL_MODEL           = _cfg.get("owl_model", "google/owlv2-large-patch14-ensemble")


# █▀▀ █▀▀▄ █▀▀▄ 　 █▀▀█ █▀▀ 　 █▀▀ █▀▀█ █▀▀▄ █▀▀ ░▀░ █▀▀▀ 
# █▀▀ █░░█ █░░█ 　 █░░█ █▀▀ 　 █░░ █░░█ █░░█ █▀▀ ▀█▀ █░▀█  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
# ▀▀▀ ▀░░▀ ▀▀▀░ 　 ▀▀▀▀ ▀░░ 　 ▀▀▀ ▀▀▀▀ ▀░░▀ ▀░░ ▀▀▀ ▀▀▀▀

# ── general functions ──────────────────────────────────────────────────────────
def CIP(items: list) -> str:
    """Converts item list into a GroundingDINO caption string."""
    return " . ".join(items) + " ."

def check(camera_feed_path):
    """Returns True if the camera feed folder is empty (or missing)."""
    if not os.path.isdir(camera_feed_path):
        return True
    with os.scandir(camera_feed_path) as it:
        return not any(it)

def create_dataset_dirs(output_path: str, annotated: bool):
    """Creates dataset directory with a raw/ folder for the JSON. Returns (dataset_name, dataset_path)."""
    now          = datetime.now().strftime('%Y-%m-%d')
    dataset_name = f"dataset_{len(os.listdir(output_path)) if os.path.isdir(output_path) else 0}_{now}".replace(" ", "_")
    dataset_path = os.path.join(output_path, dataset_name)
    os.makedirs(os.path.join(dataset_path, "raw"),    exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "labels"), exist_ok=True)
    if annotated:
        os.makedirs(os.path.join(dataset_path, "annotated"), exist_ok=True)
    return dataset_name, dataset_path

def BatchLabel(SelectedBatches: list, camera_feed: str) -> list:
    """Gathers paths for all selected image batches."""
    paths = []
    if not os.path.isdir(camera_feed):
        print(f"Camera feed path does not exist: {camera_feed}")
        return paths
    for batch in os.listdir(camera_feed):
        if batch in SelectedBatches or "all" in SelectedBatches:
            batch_path = os.path.join(camera_feed, batch)
            if os.path.isdir(batch_path):
                print(f"{batch} found")
                paths.append(batch_path)
    if len(paths) == 0:
        print(f"Cannot find any batches matching {SelectedBatches}")
    return paths

# ── JSON functions ─────────────────────────────────────────────────────────────
def build_coco_json(model_name: str) -> dict:
    """Returns an empty COCO-style inference JSON skeleton."""
    return {
        "meta": {
            "created":        datetime.now().isoformat(),
            "model":          model_name,
            "prompt":         items,
            "box_threshold":  BOX_TRESHOLD,
            "text_threshold": TEXT_TRESHOLD
        },
        "categories":  [{"id": i, "name": label} for i, label in enumerate(items)],
        "images":      [],
        "annotations": []
    }

def save_coco_json(coco: dict, dataset_path: str) -> str:
    """Saves the COCO JSON to raw/inference.json. Returns the path."""
    json_path = os.path.join(dataset_path, "raw", "inference.json")
    with open(json_path, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"Saved inference JSON -> {json_path}")
    return json_path

def JSONtoYOLO(json_path: str, dataset_path: str):
    """
    Reads a COCO-style inference JSON and writes:
      - images/    -> copied source frames
      - labels/    -> YOLO-format .txt files  (<class_id> cx cy w h, normalised)
      - annotated/ -> preview images with drawn boxes (if folder exists)
      - data.yaml  -> YOLO training config
    """
    with open(json_path, "r") as f:
        coco = json.load(f)

    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    images_map = {img["id"]: img for img in coco["images"]}

    from collections import defaultdict
    ann_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        ann_by_image[ann["image_id"]].append(ann)

    images_dir    = os.path.join(dataset_path, "images")
    labels_dir    = os.path.join(dataset_path, "labels")
    annotated_dir = os.path.join(dataset_path, "annotated")
    do_annotate   = os.path.isdir(annotated_dir)

    for img_id, img_info in tqdm(images_map.items(), desc="Exporting YOLO labels"):
        src_path = img_info["source_path"]
        fname    = img_info["file_name"]
        stem     = os.path.splitext(fname)[0]

        dest_img = os.path.join(images_dir, fname)
        if not os.path.exists(dest_img):
            shutil.copy2(src_path, dest_img)

        anns       = ann_by_image.get(img_id, [])
        yolo_lines = [
            f"{ann['category_id']} {ann['bbox'][0]:.6f} {ann['bbox'][1]:.6f} {ann['bbox'][2]:.6f} {ann['bbox'][3]:.6f}"
            for ann in anns
        ]
        with open(os.path.join(labels_dir, f"{stem}.txt"), "w") as lf:
            lf.write("\n".join(yolo_lines))

        if do_annotate and anns:
            frame = cv2.imread(src_path)
            h, w  = frame.shape[:2]
            for ann in anns:
                cx, cy, bw, bh = ann["bbox"]
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                label = categories[ann["category_id"]]
                conf  = ann["confidence"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imwrite(os.path.join(annotated_dir, fname), frame)

    yaml_data = {
        "path":  dataset_path,
        "train": "images",
        "val":   "images",
        "names": {i: cat["name"] for i, cat in enumerate(coco["categories"])}
    }
    with open(os.path.join(dataset_path, "data.yaml"), "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

    print(f"YOLO export complete -> {dataset_path}")
    print(f"  Images   : {images_dir}")
    print(f"  Labels   : {labels_dir}")
    if do_annotate:
        print(f"  Annotated: {annotated_dir}")

def empty_dataset_check(dataset_path):
    """Deletes the dataset folder if its annotated/ preview folder ended up empty."""
    annotated_dir = os.path.join(dataset_path, "annotated")
    if not os.path.isdir(annotated_dir) or len(os.listdir(annotated_dir)) == 0:
        print("dataset empty, deleting it")
        shutil.rmtree(dataset_path, ignore_errors=True)
    else:
        print("dataset fulfills check")

def print_dataset_stats(dataset_path: str):
    """Prints total images, classes, and bounding box counts for a finished dataset."""
    json_path = os.path.join(dataset_path, "raw", "inference.json")
    if not os.path.exists(json_path):
        print("[WARN]  No inference.json found, cannot print stats.")
        return

    with open(json_path) as f:
        coco = json.load(f)

    from collections import Counter
    cat_names   = {c["id"]: c["name"] for c in coco["categories"]}
    box_counts  = Counter(ann["category_id"] for ann in coco["annotations"])
    total_boxes = sum(box_counts.values())

    print("\n── Dataset Summary " + "─" * 40)
    print(f"  Dataset path : {dataset_path}")
    print(f"  Total images : {len(coco['images'])}")
    print(f"  Classes      : {len(cat_names)}")
    for cat_id, name in cat_names.items():
        print(f"    [{cat_id}] {name:30s} {box_counts.get(cat_id, 0)} boxes")
    print(f"  Total boxes  : {total_boxes}")
    print("─" * 59)

def prompt_manual_labelling(dataset_path: str):
    """After auto-labelling, optionally walk the user through manually
    reviewing/correcting each image's labels via simple CLI input."""
    images_dir = os.path.join(dataset_path, "images")
    labels_dir = os.path.join(dataset_path, "labels")

    if not os.path.isdir(images_dir):
        return

    answer = input("\nManually review/relabel this dataset now? [y/N]: ").strip().lower()
    if answer != "y":
        print("[INFO]  Skipping manual review.")
        return

    print("\nManual labelling — for each image, enter lines as:")
    print("  <class_id> <cx> <cy> <w> <h>   (normalised 0-1, space separated)")
    print("Leave blank and press Enter on its own to keep existing labels.")
    print("Type 'skip' to move to the next image without changes.")
    print("Type 'done' at any time to stop reviewing.\n")

    image_files = sorted(
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    for fname in image_files:
        stem       = os.path.splitext(fname)[0]
        label_path = os.path.join(labels_dir, f"{stem}.txt")
        existing   = ""
        if os.path.exists(label_path):
            with open(label_path) as lf:
                existing = lf.read().strip()

        print(f"\n[{fname}]")
        if existing:
            print(f"  Current labels:\n    " + existing.replace("\n", "\n    "))
        else:
            print("  Current labels: (none)")

        new_lines = []
        while True:
            line = input("  > ").strip()
            if line.lower() == "done":
                print("[INFO]  Stopping manual review.")
                return
            if line.lower() == "skip" or line == "":
                break
            new_lines.append(line)

        if new_lines:
            with open(label_path, "w") as lf:
                lf.write("\n".join(new_lines) + "\n")
            print(f"  [INFO]  Updated {label_path}")

    print("[INFO]  Manual review complete.")

# ── DINO ───────────────────────────────────────────────────────────────────────
def LabellerDino(batches, dino_type, dataset_path):
    sys.path.insert(0, GROUNDING_DINO_DIR)
    from groundingdino.util.inference import load_model, load_image, predict
    import torch.nn.functional as F
    from torchvision.ops import nms as torchvision_nms

    DINO_SCALES  = [800, 1000]
    DINO_NMS_IOU = 0.3

    def _resize_dino_tensor(image: torch.Tensor, short_edge: int) -> torch.Tensor:
        """Resizes a CHW float tensor so its shorter edge == short_edge."""
        _, h, w = image.shape
        scale   = short_edge / min(h, w)
        new_h   = int(round(h * scale))
        new_w   = int(round(w * scale))
        return F.interpolate(image.unsqueeze(0), size=(new_h, new_w),
                             mode="bilinear", align_corners=False).squeeze(0)

    types = ["SwinT", "SwinB"]
    if dino_type not in types:
        print("Dino type not found")
        return None

    if dino_type == "SwinT":
        model = load_model(
            os.path.join(GROUNDING_DINO_DIR, "groundingdino/config/GroundingDINO_SwinT_OGC.py"),
            os.path.join(GROUNDING_DINO_DIR, "weights/groundingdino_swint_ogc.pth")
        )
    else:
        model = load_model(
            os.path.join(GROUNDING_DINO_DIR, "groundingdino/config/GroundingDINO_SwinB_cfg.py"),
            os.path.join(GROUNDING_DINO_DIR, "weights/groundingdino_swinb_cogcoor.pth")
        )

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    coco     = build_coco_json(model_name=dino_type)
    caption  = CIP(items)
    image_id = 0
    ann_id   = 0

    for batch in batches:
        batch_name = os.path.basename(batch)
        images = [f for f in sorted(os.listdir(batch)) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        for fname in tqdm(images, desc=f"Labelling [{batch_name}] [Dino:{dino_type}]"):
            src_path            = os.path.join(batch, fname)
            image_source, image = load_image(src_path)
            h, w                = image_source.shape[:2]

            all_boxes, all_logits, all_phrases = [], [], []
            for scale_px in DINO_SCALES:
                image_resized = _resize_dino_tensor(image, scale_px)
                with torch.cuda.amp.autocast():
                    boxes, logits, phrases = predict(
                        model=model, image=image_resized, caption=caption,
                        box_threshold=BOX_TRESHOLD, text_threshold=TEXT_TRESHOLD, device=device
                    )
                if boxes is not None and len(boxes):
                    all_boxes.append(boxes)
                    all_logits.append(logits)
                    all_phrases.extend(phrases)

            if all_boxes:
                merged_boxes  = torch.cat(all_boxes,  dim=0)
                merged_logits = torch.cat(all_logits, dim=0)
                cx, cy, bw, bh = merged_boxes[:,0], merged_boxes[:,1], merged_boxes[:,2], merged_boxes[:,3]
                xyxy = torch.stack([cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2], dim=1)
                keep    = torchvision_nms(xyxy, merged_logits, iou_threshold=DINO_NMS_IOU)
                boxes   = merged_boxes[keep]
                logits  = merged_logits[keep]
                phrases = [all_phrases[i] for i in keep.tolist()]
            else:
                boxes, logits, phrases = [], [], []

            coco["images"].append({
                "id": image_id, "file_name": fname,
                "source_path": src_path, "width": w, "height": h
            })

            if len(phrases) > 0:
                for box, phrase, conf in zip(boxes, phrases, logits):
                    cx, cy, bw, bh = box.tolist()
                    matched = [c for c in items if c.lower() in phrase.lower()]
                    if not matched:
                        continue
                    coco["annotations"].append({
                        "id":           ann_id,
                        "image_id":     image_id,
                        "category_id":  items.index(matched[0]),
                        "bbox":         [round(cx,6), round(cy,6), round(bw,6), round(bh,6)],
                        "confidence":   round(float(conf), 6),
                        "source_model": f"{dino_type}_multiscale"
                    })
                    ann_id += 1
            image_id += 1

    json_path = os.path.join(dataset_path, "raw", "inference_dino.json")
    with open(json_path, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"Saved Dino inference JSON -> {json_path}")
    del model
    torch.cuda.empty_cache()
    return json_path

# ── Florence-2 ─────────────────────────────────────────────────────────────────
def LabellerFlorence(batches, dataset_path):
    from transformers import AutoProcessor, AutoModelForCausalLM

    FLORENCE_PROMPTS = [
        ("<OPEN_VOCABULARY_DETECTION>",   "<OPEN_VOCABULARY_DETECTION>"),
        ("<CAPTION_TO_PHRASE_GROUNDING>", "<CAPTION_TO_PHRASE_GROUNDING>"),
    ]

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(FLORENCE_PATH, trust_remote_code=True)
    model     = AutoModelForCausalLM.from_pretrained(
        FLORENCE_PATH, trust_remote_code=True,
        attn_implementation="eager", torch_dtype=torch.float16
    ).to(device)
    model.eval()

    coco     = build_coco_json(model_name="Florence-2-large")
    image_id = 0
    ann_id   = 0

    all_prompts = []
    for item in items:
        for task_prefix, task_key in FLORENCE_PROMPTS:
            prompt_text = (f"{task_prefix}{item}"
                           if task_prefix == "<OPEN_VOCABULARY_DETECTION>"
                           else f"{task_prefix}a photo of a {item}")
            all_prompts.append((prompt_text, task_key, item))

    for batch in batches:
        batch_name = os.path.basename(batch)
        images = [f for f in sorted(os.listdir(batch)) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        for fname in tqdm(images, desc=f"Labelling [{batch_name}] [Florence-2]"):
            src_path  = os.path.join(batch, fname)
            pil_image = Image.open(src_path).convert("RGB")
            w, h      = pil_image.size
            detections = []

            for prompt_text, task_key, item in all_prompts:
                inputs = processor(
                    text=prompt_text, images=pil_image,
                    return_tensors="pt"
                ).to(device)
                inputs["pixel_values"] = inputs["pixel_values"].half()

                with torch.inference_mode():
                    output_ids = model.generate(**inputs, max_new_tokens=512)

                decoded_out = processor.batch_decode(output_ids, skip_special_tokens=False)[0]
                result = processor.post_process_generation(
                    decoded_out, task=task_key, image_size=(w, h)
                )
                bboxes = result.get(task_key, {}).get("bboxes", [])
                scores = result.get(task_key, {}).get("scores", [1.0] * len(bboxes))
                for bbox, score in zip(bboxes, scores):
                    if score < FLORENCE_SCORE_THRESHOLD:
                        continue
                    x1, y1, x2, y2 = bbox
                    cx = ((x1+x2)/2) / w
                    cy = ((y1+y2)/2) / h
                    bw = (x2-x1) / w
                    bh = (y2-y1) / h
                    detections.append((item, cx, cy, bw, bh, score))

            coco["images"].append({
                "id": image_id, "file_name": fname,
                "source_path": src_path, "width": w, "height": h
            })
            for item, cx, cy, bw, bh, score in detections:
                coco["annotations"].append({
                    "id":           ann_id,
                    "image_id":     image_id,
                    "category_id":  items.index(item),
                    "bbox":         [round(cx,6), round(cy,6), round(bw,6), round(bh,6)],
                    "confidence":   score,
                    "source_model": "Florence-2-large"
                })
                ann_id += 1
            image_id += 1

    del model
    torch.cuda.empty_cache()
    return save_coco_json(coco, dataset_path)

# ── YOLO-World ─────────────────────────────────────────────────────────────────
def LabellerYOLOWorld(batches, dataset_path):
    from ultralytics import YOLOWorld
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = YOLOWorld(YOLO_WORLD_MODEL)
    model.set_classes(items)
    model.to(device)

    coco     = build_coco_json(model_name=YOLO_WORLD_MODEL)
    image_id = 0
    ann_id   = 0

    for batch in batches:
        batch_name = os.path.basename(batch)
        images = [f for f in sorted(os.listdir(batch)) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        for i in tqdm(range(0, len(images), YOLO_BATCH_SIZE), desc=f"Labelling [{batch_name}] [YOLO-World]"):
            batch_fnames = images[i : i + YOLO_BATCH_SIZE]
            src_paths    = [os.path.join(batch, f) for f in batch_fnames]
            results      = model(src_paths, conf=BOX_TRESHOLD, verbose=False)

            for fname, result in zip(batch_fnames, results):
                src_path = os.path.join(batch, fname)
                h, w     = result.orig_shape
                coco["images"].append({
                    "id": image_id, "file_name": fname,
                    "source_path": src_path, "width": w, "height": h
                })
                for box in result.boxes:
                    cls_idx  = int(box.cls)
                    cls_name = items[cls_idx] if cls_idx < len(items) else None
                    if cls_name is None:
                        continue
                    conf = float(box.conf)
                    x1n, y1n, x2n, y2n = box.xyxyn[0].tolist()
                    cx = (x1n+x2n) / 2
                    cy = (y1n+y2n) / 2
                    bw = x2n - x1n
                    bh = y2n - y1n
                    coco["annotations"].append({
                        "id":           ann_id,
                        "image_id":     image_id,
                        "category_id":  cls_idx,
                        "bbox":         [round(cx,6), round(cy,6), round(bw,6), round(bh,6)],
                        "confidence":   round(conf, 6),
                        "source_model": YOLO_WORLD_MODEL
                    })
                    ann_id += 1
                image_id += 1

    json_path = os.path.join(dataset_path, "raw", "inference_yoloworld.json")
    with open(json_path, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"Saved YOLO-World inference JSON -> {json_path}")
    del model
    torch.cuda.empty_cache()
    return json_path

# ── OWLv2 ──────────────────────────────────────────────────────────────────────
def LabellerOWL(batches, dataset_path):
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Owlv2Processor.from_pretrained(OWL_MODEL)
    model     = Owlv2ForObjectDetection.from_pretrained(
        OWL_MODEL, torch_dtype=torch.float16
    ).to(device)
    model.eval()

    owl_queries = [f"a photo of a {it}" for it in items]
    coco        = build_coco_json(model_name=OWL_MODEL)
    image_id    = 0
    ann_id      = 0

    for batch in batches:
        batch_name = os.path.basename(batch)
        images = [f for f in sorted(os.listdir(batch)) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        for i in tqdm(range(0, len(images), OWL_BATCH_SIZE), desc=f"Labelling [{batch_name}] [OWLv2]"):
            batch_fnames = images[i : i + OWL_BATCH_SIZE]
            pil_images   = [Image.open(os.path.join(batch, f)).convert("RGB") for f in batch_fnames]
            sizes        = [img.size for img in pil_images]   # list of (orig_w, orig_h)

            inputs = processor(
                text=[owl_queries] * len(pil_images),
                images=pil_images,
                return_tensors="pt",
                padding=True
            ).to(device)

            with torch.inference_mode():
                outputs = model(**inputs)

            target_sizes = torch.tensor(
                [[orig_h, orig_w] for orig_w, orig_h in sizes], device=device
            )
            batch_results = processor.post_process_object_detection(
                outputs, threshold=OWL_SCORE_THRESHOLD, target_sizes=target_sizes
            )

            for fname, results, (orig_w, orig_h) in zip(batch_fnames, batch_results, sizes):
                src_path = os.path.join(batch, fname)
                coco["images"].append({
                    "id": image_id, "file_name": fname,
                    "source_path": src_path, "width": orig_w, "height": orig_h
                })
                for box, score, lbl in zip(results["boxes"], results["scores"], results["labels"]):
                    score = float(score)
                    x1, y1, x2, y2 = box.tolist()
                    cx = ((x1+x2)/2) / orig_w
                    cy = ((y1+y2)/2) / orig_h
                    bw = (x2-x1) / orig_w
                    bh = (y2-y1) / orig_h
                    coco["annotations"].append({
                        "id":           ann_id,
                        "image_id":     image_id,
                        "category_id":  int(lbl),
                        "bbox":         [round(cx,6), round(cy,6), round(bw,6), round(bh,6)],
                        "confidence":   round(score, 6),
                        "source_model": OWL_MODEL
                    })
                    ann_id += 1
                image_id += 1

    json_path = os.path.join(dataset_path, "raw", "inference_owl.json")
    with open(json_path, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"Saved OWLv2 inference JSON -> {json_path}")
    del model
    torch.cuda.empty_cache()
    return json_path

# ── YOLO ───────────────────────────────────────────────────────────────────────
def LabellerYOLO(batches, dataset_path):
    from ultralytics import YOLO

    if not YOLO_MODEL_PATH:
        print("[ERROR] 'yolo_model_path' not set in config, skipping YOLO labeller.")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = YOLO(YOLO_MODEL_PATH)
    model.to(device)

    coco     = build_coco_json(model_name=os.path.basename(YOLO_MODEL_PATH))
    image_id = 0
    ann_id   = 0

    for batch in batches:
        batch_name = os.path.basename(batch)
        images = [f for f in sorted(os.listdir(batch)) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        for i in tqdm(range(0, len(images), YOLO_BATCH_SIZE), desc=f"Labelling [{batch_name}] [YOLO]"):
            batch_fnames = images[i : i + YOLO_BATCH_SIZE]
            src_paths    = [os.path.join(batch, f) for f in batch_fnames]
            results      = model(src_paths, conf=BOX_TRESHOLD, verbose=False)

            for fname, result in zip(batch_fnames, results):
                src_path = os.path.join(batch, fname)
                h, w     = result.orig_shape
                coco["images"].append({
                    "id": image_id, "file_name": fname,
                    "source_path": src_path, "width": w, "height": h
                })
                for box in result.boxes:
                    cls_name = model.names[int(box.cls)]
                    matched  = [it for it in items if it.lower() in cls_name.lower() or cls_name.lower() in it.lower()]
                    if not matched:
                        continue
                    conf = float(box.conf)
                    x1n, y1n, x2n, y2n = box.xyxyn[0].tolist()
                    cx = (x1n+x2n) / 2
                    cy = (y1n+y2n) / 2
                    bw = x2n - x1n
                    bh = y2n - y1n
                    coco["annotations"].append({
                        "id":           ann_id,
                        "image_id":     image_id,
                        "category_id":  items.index(matched[0]),
                        "bbox":         [round(cx,6), round(cy,6), round(bw,6), round(bh,6)],
                        "confidence":   round(conf, 6),
                        "source_model": os.path.basename(YOLO_MODEL_PATH)
                    })
                    ann_id += 1
                image_id += 1

    json_path = os.path.join(dataset_path, "raw", "inference_yolo.json")
    with open(json_path, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"Saved YOLO inference JSON -> {json_path}")
    del model
    torch.cuda.empty_cache()
    return json_path

# ── Merge + NMS ────────────────────────────────────────────────────────────────
def MergeCocoJsons(json_paths: list, dataset_path: str) -> str:
    """
    Merges multiple COCO-style inference JSONs into one raw/inference.json.
    Re-indexes image_id and ann_id. Images are keyed on file_name so
    annotations from different models on the same frame share one image record.
    Runs per-image NMS per category to remove duplicate boxes across models.
    """
    from collections import defaultdict
    from torchvision.ops import nms as torchvision_nms

    with open(json_paths[0]) as f:
        base = json.load(f)
    categories  = base["categories"]

    fname_to_id   = {}
    merged_images = []
    raw_anns      = []
    next_img_id   = 0

    for json_path in json_paths:
        with open(json_path) as f:
            coco = json.load(f)
        old_to_new = {}
        for img in coco["images"]:
            fname = img["file_name"]
            if fname not in fname_to_id:
                fname_to_id[fname] = next_img_id
                merged_images.append({**img, "id": next_img_id})
                next_img_id += 1
            old_to_new[img["id"]] = fname_to_id[fname]
        for ann in coco["annotations"]:
            raw_anns.append({**ann, "image_id": old_to_new[ann["image_id"]]})

    groups = defaultdict(list)
    for ann in raw_anns:
        groups[(ann["image_id"], ann["category_id"])].append(ann)

    merged_anns = []
    next_ann_id = 0
    for (img_id, cat_id), anns in groups.items():
        if len(anns) == 1:
            merged_anns.append({**anns[0], "id": next_ann_id})
            next_ann_id += 1
            continue
        boxes  = []
        scores = []
        for ann in anns:
            cx, cy, bw, bh = ann["bbox"]
            boxes.append([cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2])
            scores.append(ann["confidence"])
        boxes_t  = torch.tensor(boxes,  dtype=torch.float32)
        scores_t = torch.tensor(scores, dtype=torch.float32)
        keep     = torchvision_nms(boxes_t, scores_t, iou_threshold=MERGE_NMS_IOU)
        for i in keep.tolist():
            merged_anns.append({**anns[i], "id": next_ann_id})
            next_ann_id += 1

    merged = {
        "meta": {
            "created":        datetime.now().isoformat(),
            "model":          "merged+nms:" + "+".join(os.path.basename(p) for p in json_paths),
            "prompt":         items,
            "box_threshold":  BOX_TRESHOLD,
            "text_threshold": TEXT_TRESHOLD,
            "merge_nms_iou":  MERGE_NMS_IOU
        },
        "categories":  categories,
        "images":      merged_images,
        "annotations": merged_anns
    }

    out_path = os.path.join(dataset_path, "raw", "inference.json")
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)

    before = len(raw_anns)
    after  = len(merged_anns)
    print(f"Merged {len(json_paths)} JSONs -> {out_path}")
    print(f"  Annotations: {before} raw -> {after} after NMS ({before - after} duplicates removed)")
    return out_path

# ── main ───────────────────────────────────────────────────────────────────────
def main():
    if check(CAMERA_FEED_PATH):
        print("Camera feed is empty, nothing to label.")
        return None

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    dataset_name, dataset_path = create_dataset_dirs(OUTPUT_PATH, annotated)
    print(f"Dataset directory: {dataset_path}")

    batches = BatchLabel(BATCH_SELECTION, CAMERA_FEED_PATH)

    if not batches:
        return dataset_path

    json_paths = []

    if "DINO" in models2run:
        json_path_dino = LabellerDino(batches, DINO_TYPE, dataset_path)
        if json_path_dino:
            json_paths.append(json_path_dino)

    if "FLORANCE" in models2run:
        json_path_florence = LabellerFlorence(batches, dataset_path)
        if json_path_florence:
            json_paths.append(json_path_florence)

    if "YOLO" in models2run:
        json_path_yolo = LabellerYOLO(batches, dataset_path)
        if json_path_yolo:
            json_paths.append(json_path_yolo)

    if "YOLOWORLD" in models2run:
        json_path_yoloworld = LabellerYOLOWorld(batches, dataset_path)
        if json_path_yoloworld:
            json_paths.append(json_path_yoloworld)

    if "OWL2" in models2run:
        json_path_owl = LabellerOWL(batches, dataset_path)
        if json_path_owl:
            json_paths.append(json_path_owl)

    if not json_paths:
        print("All labellers failed, nothing to export.")
        return dataset_path

    merged_path = MergeCocoJsons(json_paths, dataset_path) if len(json_paths) > 1 else json_paths[0]
    JSONtoYOLO(merged_path, dataset_path)
    print(f"\nDone. Dataset ready at: {dataset_path}")

    print_dataset_stats(dataset_path)
    prompt_manual_labelling(dataset_path)

    return dataset_path

if __name__ == "__main__":
    _result_dataset_path = None
    try:
        _result_dataset_path = main()
    finally:
        print("program closed..")
        if _result_dataset_path and os.path.isdir(_result_dataset_path):
            print("checking dataset")
            empty_dataset_check(_result_dataset_path)
        else:
            print("no dataset was created, skipping check")