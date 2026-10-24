"""Train once into fixed runs/training and runs/weights directories."""
from pathlib import Path
import hashlib
import shutil
from .config import RUNS, load_config, project_path, write_yaml, require_file
from .dataset import dataset_yaml

def train():
    config = load_config("train")
    data = dataset_yaml(load_config("dataset"))
    if (RUNS / "training/results.csv").exists():
        raise ValueError("Training results already exist in runs/training. This project uses one fixed training run.")
    from ultralytics import YOLO
    from ultralytics.utils.downloads import attempt_download_asset
    pretrained = project_path(config.pop("pretrained"))
    pretrained.parent.mkdir(parents=True, exist_ok=True)
    if not pretrained.is_file():
        print(f"Downloading official pretrained weights: {pretrained}", flush=True)
        attempt_download_asset(str(pretrained))
    require_file(pretrained)
    model = YOLO(str(pretrained))
    model.train(data=str(data), project=str(RUNS), name="training", exist_ok=True, **config)
    weights = RUNS / "weights"
    weights.mkdir(parents=True, exist_ok=True)
    for name in ("best.pt", "last.pt"):
        source = require_file(Path(model.trainer.save_dir) / "weights" / name)
        shutil.move(str(source), str(weights / name))
    source_dir = Path(model.trainer.save_dir) / "weights"
    if not any(source_dir.iterdir()):
        source_dir.rmdir()
    curves = RUNS / "training/curves"
    for image in list((RUNS / "training").glob("*.png")) + list((RUNS / "training").glob("*.jpg")):
        curves.mkdir(exist_ok=True)
        shutil.move(str(image), str(curves / image.name))
    write_yaml(RUNS / "training/config.yaml", {"train": load_config("train"), "dataset": load_config("dataset")})
    write_yaml(weights / "training.yaml", {"best_sha256": hashlib.sha256((weights / "best.pt").read_bytes()).hexdigest()})
    print(f"Weights: {weights}; results: {RUNS / 'training/results.csv'}")
