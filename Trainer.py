from ultralytics import YOLO, settings
from ultralytics.data.split import autosplit
from pathlib import Path
from datetime import date
import argparse
import os
import shutil
import json
import subprocess
import yaml

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Generalized YOLO trainer")
parser.add_argument("--config", default=None,
                    help="Path to a JSON config file (see below for keys)")
args = parser.parse_args()

# ── Load config (file path) or fall back to script-level defaults ────────────
# Required config keys: datasets_path, dataset_names, base_model, imgsz,
#                        epochs, batch, rearrange
# Optional config keys: distribution, output_path, workers, device,
#                        optimizer, lr0, lrf, patience
if args.config and os.path.isfile(args.config):
    with open(args.config, "r") as f:
        _cfg = json.load(f)
    print(f"[INFO]  Loaded config from {args.config}")

    BaseDir        = os.path.dirname(os.path.abspath(__file__))
    dataset_folder  = _cfg["datasets_path"]
    output_path     = _cfg.get("output_path", os.path.join(BaseDir, "trained_models"))

    dataset_names   = _cfg["dataset_names"]
    modelUsed       = _cfg["base_model"]
    image_size      = int(_cfg["imgsz"])
    epoches         = int(_cfg["epochs"])
    batch           = int(_cfg["batch"])
    rearrange       = bool(_cfg["rearrange"])
    distribution    = tuple(_cfg.get("distribution", (0.8, 0.1, 0.1)))
    workers         = int(_cfg.get("workers", 8))
    device          = _cfg.get("device", "cuda")
    optimizer       = _cfg.get("optimizer", "AdamW")
    lr0             = float(_cfg.get("lr0", 0.001))
    lrf             = float(_cfg.get("lrf", 0.01))
    patience        = int(_cfg.get("patience", 50))


# ██████╗░██████╗░░█████╗░░██████╗░██████╗░░█████╗░███╗░░░███╗  ░█████╗░░█████╗░███╗░░██╗███████╗██╗░██████╗░
# ██╔══██╗██╔══██╗██╔══██╗██╔════╝░██╔══██╗██╔══██╗████╗░████║  ██╔══██╗██╔══██╗████╗░██║██╔════╝██║██╔════╝░
# ██████╔╝██████╔╝██║░░██║██║░░██╗░██████╔╝███████║██╔████╔██║  ██║░░╚═╝██║░░██║██╔██╗██║█████╗░░██║██║░░██╗░
# ██╔═══╝░██╔══██╗██║░░██║██║░░╚██╗██╔══██╗██╔══██║██║╚██╔╝██║  ██║░░██╗██║░░██║██║╚████║██╔══╝░░██║██║░░╚██╗
# ██║░░░░░██║░░██║╚█████╔╝╚██████╔╝██║░░██║██║░░██║██║░╚═╝░██║  ╚█████╔╝╚█████╔╝██║░╚███║██║░░░░░██║╚██████╔╝
# ╚═╝░░░░░╚═╝░░╚═╝░╚════╝░░╚═════╝░╚═╝░░╚═╝╚═╝░░╚═╝╚═╝░░░░░╚═╝  ░╚════╝░░╚════╝░╚═╝░░╚══╝╚═╝░░░░░╚═╝░╚═════╝░
# edit here to change config for trainer
else:
    # ── Standalone defaults — edit these for a quick manual run ─────────────
    print("[WARN]  No --config supplied; using script-level defaults.")
    BaseDir        = os.path.dirname(os.path.abspath(__file__))
    dataset_folder = os.path.join(BaseDir, "Trainer-Datasets")      # place to store datasets
    output_path    = os.path.join(BaseDir, "Trainer-TrainedModels") # output for where models will be saved

    dataset_names  = ["MyDataset1", "MyDataset2"]
    modelUsed      = "yolo11n.pt"
    image_size     = 640
    epoches        = 100
    batch          = -1
    distribution   = (0.8, 0.1, 0.1)
    rearrange      = True
    workers        = 8
    device         = "cuda"
    optimizer      = "AdamW"
    lr0            = 0.001
    lrf            = 0.01
    patience       = 50

basename = modelUsed.split(".")[0]


# ── Functions ─────────────────────────────────────────────────────────────────

def max_gpu():
    """Optionally lock GPU clocks for consistent training throughput. Safe no-op on failure."""
    cmds = [
        "sudo nvidia-smi -pm 1",
        "sudo nvidia-smi --auto-boost-default=0",
    ]
    for cmd in cmds:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[INFO]  done: {cmd}")
        else:
            print(f"[WARN]  skipped: {cmd}: {result.stderr.strip()}")


def get_dataset_path(dataset_name: str) -> str:
    return os.path.join(dataset_folder, dataset_name)


def load_yaml_classes(dataset_path: str) -> dict:
    yaml_path = os.path.join(dataset_path, "data.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"No data.yaml found in {dataset_path}")
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    names = data.get("names", {})
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return {int(k): v for k, v in names.items()}


def merge_datasets(dataset_names: list, dataset_folder: str) -> str:
    """Merge one or more YOLO-format datasets into a unified class space.
    Single-dataset runs skip merging entirely."""
    if len(dataset_names) == 1:
        print(f"[INFO]  Single dataset mode: {dataset_names[0]}")
        return get_dataset_path(dataset_names[0])

    merged_path   = os.path.join(dataset_folder, "_merged_temp")
    merged_images = os.path.join(merged_path, "images")
    merged_labels = os.path.join(merged_path, "labels")
    os.makedirs(merged_images, exist_ok=True)
    os.makedirs(merged_labels, exist_ok=True)

    print(f"[INFO]  Merging {len(dataset_names)} datasets into {merged_path} ...")

    unified_classes = {}
    dataset_remaps  = []

    for ds_name in dataset_names:
        ds_path       = get_dataset_path(ds_name)
        local_classes = load_yaml_classes(ds_path)
        remap = {}
        for local_id, class_name in local_classes.items():
            if class_name not in unified_classes:
                unified_classes[class_name] = len(unified_classes)
            remap[local_id] = unified_classes[class_name]
        dataset_remaps.append(remap)
        print(f"[INFO]    {ds_name}: {len(local_classes)} classes -> remap {remap}")

    unified_names = {v: k for k, v in unified_classes.items()}
    print(f"[INFO]  Unified class space: {len(unified_classes)} classes - {list(unified_classes.keys())}")

    for ds_name, remap in zip(dataset_names, dataset_remaps):
        ds_path   = get_dataset_path(ds_name)
        img_dir   = os.path.join(ds_path, "images")
        label_dir = os.path.join(ds_path, "labels")

        for img_file in Path(img_dir).iterdir():
            if not img_file.is_file():
                continue
            dest_name  = f"{ds_name}__{img_file.name}"
            dest_img   = os.path.join(merged_images, dest_name)
            if not os.path.exists(dest_img):
                os.symlink(str(img_file), dest_img)

            label_file = Path(label_dir) / (img_file.stem + ".txt")
            dest_label = os.path.join(merged_labels, dest_name.rsplit(".", 1)[0] + ".txt")
            if label_file.exists():
                with open(label_file, "r") as f:
                    lines = f.readlines()
                remapped_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    local_id   = int(parts[0])
                    unified_id = remap.get(local_id, local_id)
                    remapped_lines.append(f"{unified_id} " + " ".join(parts[1:]))
                with open(dest_label, "w") as f:
                    f.write("\n".join(remapped_lines) + "\n")
            else:
                open(dest_label, "w").close()

    yaml_data = {
        "path":  merged_path,
        "train": "autosplit_train.txt",
        "val":   "autosplit_val.txt",
        "test":  "autosplit_test.txt",
        "nc":    len(unified_classes),
        "names": [unified_names[i] for i in sorted(unified_names)],
    }
    with open(os.path.join(merged_path, "data.yaml"), "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

    print(f"[INFO]  Merged dataset ready: {merged_path}")
    return merged_path


def cleanup_merged(dataset_path: str, dataset_names: list):
    if len(dataset_names) > 1 and dataset_path.endswith("_merged_temp"):
        print(f"[INFO]  Cleaning up merged dataset: {dataset_path}")
        shutil.rmtree(dataset_path, ignore_errors=True)


def prepare_splits(dataset_path: str, distribution: tuple, rearrange: bool):
    if rearrange:
        print("[INFO]  Reshuffling data splits...")
        autosplit(path=f"{dataset_path}/images", weights=distribution, annotated_only=False)
    else:
        for f in ("autosplit_train.txt", "autosplit_val.txt", "autosplit_test.txt"):
            if not os.path.exists(f"{dataset_path}/{f}"):
                print(f"[INFO]  {f} missing - creating splits now...")
                autosplit(path=f"{dataset_path}/images", weights=distribution, annotated_only=False)
                break
        print("[INFO]  Splits check done.")


def get_next_folder_name(output_path: str) -> str:
    existing = os.listdir(output_path) if os.path.isdir(output_path) else []
    index    = len(existing)
    today    = date.today()
    return f"model_{index}_{today}"


# ███╗░░░███╗░█████╗░██████╗░███████╗██╗░░░░░  ██████╗░░█████╗░██████╗░░█████╗░███╗░░░███╗░██████╗
# ████╗░████║██╔══██╗██╔══██╗██╔════╝██║░░░░░  ██╔══██╗██╔══██╗██╔══██╗██╔══██╗████╗░████║██╔════╝
# ██╔████╔██║██║░░██║██║░░██║█████╗░░██║░░░░░  ██████╔╝███████║██████╔╝███████║██╔████╔██║╚█████╗░
# ██║╚██╔╝██║██║░░██║██║░░██║██╔══╝░░██║░░░░░  ██╔═══╝░██╔══██║██╔══██╗██╔══██║██║╚██╔╝██║░╚═══██╗
# ██║░╚═╝░██║╚█████╔╝██████╔╝███████╗███████╗  ██║░░░░░██║░░██║██║░░██║██║░░██║██║░╚═╝░██║██████╔╝
# ╚═╝░░░░░╚═╝░╚════╝░╚═════╝░╚══════╝╚══════╝  ╚═╝░░░░░╚═╝░░╚═╝╚═╝░░╚═╝╚═╝░░╚═╝╚═╝░░░░░╚═╝╚═════╝░
# edit here for additional params
def train_model(dataset_path: str, output_path: str, folder_name: str) -> YOLO:
    model = YOLO(modelUsed)
    print(f"[INFO]  Training started - model={modelUsed}  epochs={epoches}  imgsz={image_size}  batch={batch}")
    model.train(
        data=f"{dataset_path}/data.yaml",
        epochs=epoches,
        imgsz=image_size,
        batch=batch,
        workers=workers,
        amp=True,
        device=device,

        # ── Optimisation ──────────────────────────────────────────────────────
        optimizer=optimizer,
        lr0=lr0,
        lrf=lrf,

        # ── Augmentation ──────────────────────────────────────────────────────
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        close_mosaic=10,

        cache=True,
        patience=patience,

        project=f"{output_path}/{folder_name}",
    )
    return model

# █▀▀ █▀▀▄ █▀▀▄ 　 █▀▀█ █▀▀ 　 █▀▀ █▀▀█ █▀▀▄ █▀▀ ░▀░ █▀▀▀ 
# █▀▀ █░░█ █░░█ 　 █░░█ █▀▀ 　 █░░ █░░█ █░░█ █▀▀ ▀█▀ █░▀█  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
# ▀▀▀ ▀░░▀ ▀▀▀░ 　 ▀▀▀▀ ▀░░ 　 ▀▀▀ ▀▀▀▀ ▀░░▀ ▀░░ ▀▀▀ ▀▀▀▀

def evaluate_model(model: YOLO, dataset_path: str, output_path: str, folder_name: str) -> dict:
    print("[INFO]  Evaluating model...")
    metrics = model.val(
        data=f"{dataset_path}/data.yaml",
        imgsz=image_size,
        project=f"{output_path}/{folder_name}",
    )
    return {
        "map50":     metrics.box.map50,
        "map50_95":  metrics.box.map,
        "precision": metrics.box.mp,
        "recall":    metrics.box.mr,
    }


def save_model_details(output_path: str, folder_name: str, dataset_path: str,
                       stats: dict, model: YOLO, dataset_names: list):
    today   = date.today()
    details = {
        "ModelUsed":   modelUsed,
        "TimeTrained": str(today),
        "Stats":       stats,
        "DataStats": {
            "Datasets":     dataset_names,
            "DataSize":     len(os.listdir(f"{dataset_path}/images")),
            "ImgSize":      image_size,
            "Epochs":       epoches,
            "Batch":        batch,
            "Distribution": list(distribution),
        },
        "Classes": model.names,
    }
    out_file = f"{output_path}/{folder_name}/ModelDetails.json"
    with open(out_file, "w") as f:
        json.dump(details, f, indent=4)
    print(f"[DONE]  Model details saved to {out_file}")


def prompt_rename_model(output_path: str, folder_name: str) -> str:
    """Ask the user for a friendly model name and rename best.pt accordingly.
    Returns the path to the (possibly renamed) .pt file."""
    weights_dir = f"{output_path}/{folder_name}/weights"
    best_pt     = f"{weights_dir}/best.pt"

    if not os.path.exists(best_pt):
        print(f"[WARN]  {best_pt} not found, skipping rename.")
        return best_pt

    name = input("\nName this model (leave blank to keep 'best.pt'): ").strip()
    if not name:
        return best_pt

    safe_name = "".join(c for c in name if c.isalnum() or c in ("_", "-")) or "model"
    new_pt    = f"{weights_dir}/{safe_name}.pt"
    shutil.copy(best_pt, new_pt)
    print(f"[INFO]  Saved as {new_pt}")
    return new_pt


def prompt_export(pt_path: str):
    """Ask the user whether to export the trained model, and to which format(s)."""
    print("\nExport this model?")
    print("  1) Skip")
    print("  2) ONNX")
    print("  3) IMX500 (.rpk, via Sony MCT pipeline)")
    print("  4) Both ONNX and IMX500")
    choice = input("Choice [1-4]: ").strip() or "1"

    if choice == "1":
        print("[INFO]  Skipping export.")
        return

    model = YOLO(pt_path)

    if choice in ("2", "4"):
        print("[INFO]  Exporting to ONNX...")
        onnx_path = model.export(format="onnx", imgsz=image_size)
        print(f"[DONE]  ONNX saved to {onnx_path}")

    if choice in ("3", "4"):
        calib_yaml = input(
            "Path to a data.yaml for IMX500 INT8 calibration "
            "(any dataset folder containing real images works): "
        ).strip()
        if not calib_yaml or not os.path.exists(calib_yaml):
            print("[WARN]  No valid data.yaml provided, skipping IMX500 export.")
            return
        imx_imgsz = input("IMX500 export image size (e.g. 320): ").strip()
        imx_imgsz = int(imx_imgsz) if imx_imgsz.isdigit() else 320
        print(f"[INFO]  Exporting to IMX500 format at imgsz={imx_imgsz}...")
        try:
            rpk_path = model.export(format="imx", imgsz=imx_imgsz, data=calib_yaml)
            print(f"[DONE]  IMX500 model saved to {rpk_path}")
        except Exception as e:
            print(f"[ERROR] IMX500 export failed: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    max_gpu()
    settings.update({"datasets_dir": dataset_folder})
    dataset_path = merge_datasets(dataset_names, dataset_folder)
    prepare_splits(dataset_path, distribution, rearrange)
    os.makedirs(output_path, exist_ok=True)
    folder_name = get_next_folder_name(output_path)
    os.makedirs(f"{output_path}/{folder_name}", exist_ok=True)
    model  = train_model(dataset_path, output_path, folder_name)
    stats  = evaluate_model(model, dataset_path, output_path, folder_name)
    save_model_details(output_path, folder_name, dataset_path, stats, model, dataset_names)
    cleanup_merged(dataset_path, dataset_names)
    print(f"[DONE]  Training complete! Model saved to: {output_path}/{folder_name}")

    final_pt = prompt_rename_model(output_path, folder_name)
    prompt_export(final_pt)