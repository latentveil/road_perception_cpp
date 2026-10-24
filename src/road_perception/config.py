"""Resolve repository resources and DATA_ROOT-relative data paths."""
from pathlib import Path
import csv
import os
import yaml

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"

def _relative(root, value):
    path = Path(value)
    if path.is_absolute() or path.drive or path.root or ".." in path.parts:
        raise ValueError(f"Configuration paths must be relative: {value}")
    return root / path

def project_path(value):
    return _relative(ROOT, value)

def data_path(value):
    value_root = os.environ.get("DATA_ROOT")
    if not value_root:
        raise ValueError("Set DATA_ROOT first: export DATA_ROOT=<your dataset root>")
    root = Path(value_root).expanduser()
    if not root.is_absolute():
        raise ValueError("DATA_ROOT must be an absolute directory")
    if not root.is_dir():
        raise FileNotFoundError(f"DATA_ROOT directory does not exist: {root}")
    return _relative(root, value)

def load_config(name):
    with project_path(f"configs/{name}.yaml").open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in configs/{name}.yaml")
    return value

def require_file(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path

def write_yaml(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8")

def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def sequences(config):
    groups = []
    for key in ("train_sequences", "val_sequences"):
        path = require_file(project_path(config[key]))
        names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        if not names or any(Path(n).name != n or not n.startswith("MVI_") for n in names):
            raise ValueError(f"Invalid sequence list: {path}")
        groups.append(names)
    combined = groups[0] + groups[1]
    if len(combined) != len(set(combined)):
        raise ValueError("Duplicate or overlapping train/val sequences")
    if len(combined) != int(config["expected_sequences"]):
        raise ValueError(f"Expected {config['expected_sequences']} sequences, got {len(combined)}")
    return {"train": groups[0], "val": groups[1]}

def cli(action):
    try:
        action()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error))
