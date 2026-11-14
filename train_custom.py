"""
train_custom.py
----------------
Automates the "Custom Object Detection" half of the project:
1. Clones the official Ultralytics YOLOv5 repo (needed for train.py,
    which torch.hub does not expose directly).
2. Installs its requirements.
3. Writes a dataset YAML file (class names + paths) from the values you pass in.
4. Kicks off training with train.py.

Usage
-----
    python train_custom.py \
        --dataset_dir custom_dataset \
        --classes car bike autorickshaw \
        --epochs 50 --batch 16 --img 640
"""

import argparse       # Reads command-line flags
import os             # Builds file paths, checks folder existence
import subprocess     # Runs shell commands (git clone, pip install, train.py)
import sys            # Access to the current Python interpreter path
import yaml           # Writes the dataset config as a proper YAML file


YOLOV5_REPO = "https://github.com/ultralytics/yolov5.git"
YOLOV5_DIR = "yolov5"  # Local folder name the repo will be cloned into


def run(cmd: list):
    """Run a shell command, streaming output live, and stop on failure."""
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"[ERROR] Command failed: {' '.join(cmd)}")


def ensure_yolov5_repo():
    """Clone the YOLOv5 repo (once) and install its Python dependencies."""
    if not os.path.isdir(YOLOV5_DIR):
        run(["git", "clone", YOLOV5_REPO, YOLOV5_DIR])
    else:
        print(f"[INFO] '{YOLOV5_DIR}' already exists, skipping clone.")

    requirements_path = os.path.join(YOLOV5_DIR, "requirements.txt")
    run([sys.executable, "-m", "pip", "install", "-r", requirements_path])


def write_dataset_yaml(dataset_dir: str, classes: list, yaml_path: str):
    """
    Build the dataset.yaml file YOLOv5's train.py expects. It needs:
    - train / val : paths to the image folders
    - nc          : number of classes
    - names       : list of class name strings, in class_id order
    """
    dataset_dir = os.path.abspath(dataset_dir)

    config = {
        "train": os.path.join(dataset_dir, "images", "train"),
        "val": os.path.join(dataset_dir, "images", "val"),
        "nc": len(classes),
        "names": classes,
    }

    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"[INFO] Wrote dataset config to {yaml_path}:")
    print(yaml.dump(config, default_flow_style=False))


def check_dataset_structure(dataset_dir: str):
    """
    Sanity-check that the expected folders exist and are non-empty before
    we spend time launching a training run.
    """
    required = [
        os.path.join(dataset_dir, "images", "train"),
        os.path.join(dataset_dir, "images", "val"),
        os.path.join(dataset_dir, "labels", "train"),
        os.path.join(dataset_dir, "labels", "val"),
    ]
    missing_or_empty = []
    for folder in required:
        if not os.path.isdir(folder) or len(os.listdir(folder)) == 0:
            missing_or_empty.append(folder)

    if missing_or_empty:
        print("[WARNING] The following folders are missing or empty:")
        for folder in missing_or_empty:
            print(f"    - {folder}")
        print(
            "[WARNING] Add your labeled images (from LabelImg/Roboflow) "
            "before training, or the run will fail or produce a useless model."
        )


def train(yaml_path: str, epochs: int, batch: int, img: int, weights: str, name: str):
    """Launch YOLOv5's own train.py with the arguments the assignment specifies."""
    train_script = os.path.join(YOLOV5_DIR, "train.py")
    run([
        sys.executable, train_script,
        "--img", str(img),
        "--batch", str(batch),
        "--epochs", str(epochs),
        "--data", os.path.abspath(yaml_path),
        "--weights", weights,     # e.g. yolov5s.pt -> fine-tune from pretrained COCO weights
        "--name", name,
    ])
    print(
        f"[INFO] Training complete. Best weights saved to "
        f"'{YOLOV5_DIR}/runs/train/{name}/weights/best.pt'"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv5 on a custom dataset")
    parser.add_argument("--dataset_dir", type=str, default="custom_dataset",
                        help="Root folder containing images/ and labels/ subfolders")
    parser.add_argument("--classes", type=str, nargs="+", required=True,
                        help="List of class names, e.g. --classes car bike autorickshaw")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--img", type=int, default=640, help="Training image size")
    parser.add_argument("--weights", type=str, default="yolov5s.pt",
                        help="Starting weights to fine-tune from")
    parser.add_argument("--name", type=str, default="my_custom_model",
                        help="Name for this training run (used in output folder)")
    return parser.parse_args()


def main():
    args = parse_args()

    ensure_yolov5_repo()

    yaml_path = "custom.yaml"
    write_dataset_yaml(args.dataset_dir, args.classes, yaml_path)

    check_dataset_structure(args.dataset_dir)

    train(yaml_path, args.epochs, args.batch, args.img, args.weights, args.name)


if __name__ == "__main__":
    main()
