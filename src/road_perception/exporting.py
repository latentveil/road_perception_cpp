"""Export the trained checkpoint to a fixed ONNX model."""
from pathlib import Path
import hashlib
import shutil
from .config import RUNS, load_config, require_file, write_yaml

def export(cpu=False):
    config = load_config("export")
    if config.get("dynamic") or config.get("nms"):
        raise ValueError("The deployment contract requires dynamic=false and nms=false")
    if int(config["imgsz"]) != int(load_config("inference")["input_size"]):
        raise ValueError("Export imgsz must equal inference input_size")
    checkpoint = require_file(RUNS / "weights/best.pt")
    from ultralytics import YOLO
    if cpu:
        config["device"] = "cpu"
    model = YOLO(str(checkpoint))
    if list(model.names.values()) != load_config("inference")["names"]:
        raise ValueError("The trained checkpoint must use the four UA-DETRAC class names in order")
    generated = Path(model.export(format="onnx", **config))
    target = RUNS / "weights/best.onnx"
    if generated.resolve() != target.resolve():
        shutil.move(str(generated), str(target))
    write_yaml(RUNS / "weights/export.yaml", {"export": config, "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
    print(f"Exported: {target}")
