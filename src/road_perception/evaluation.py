"""Evaluate detection on the validation split and save per-class CSV results."""
from .config import RUNS, load_config, require_file, write_csv
from .dataset import dataset_yaml

def evaluate(backend="pt", cpu=False):
    from ultralytics import YOLO
    data = dataset_yaml(load_config("dataset"))
    config = load_config("inference")
    model = require_file(RUNS / "weights" / ("best.pt" if backend == "pt" else "best.onnx"))
    results = YOLO(str(model)).val(data=str(data), imgsz=config["input_size"], device="cpu" if cpu else config["device"],
        project=str(RUNS / "evaluation"), name=backend, exist_ok=True, plots=True)
    box = results.box
    rows = [{"backend": backend, "class": "all", "precision": float(box.mp), "recall": float(box.mr), "map50": float(box.map50), "map50_95": float(box.map)}]
    for index, class_id in enumerate(box.ap_class_index):
        rows.append({"backend": backend, "class": config["names"][int(class_id)], "precision": float(box.p[index]), "recall": float(box.r[index]), "map50": float(box.ap50[index]), "map50_95": float(box.ap[index])})
    target = RUNS / "evaluation" / f"detection_{backend}.csv"
    write_csv(target, rows, ["backend", "class", "precision", "recall", "map50", "map50_95"])
    # Preserve both backends when evaluated sequentially.
    import csv
    combined = []
    for path in sorted((RUNS / "evaluation").glob("detection_*.csv")):
        with path.open(newline="", encoding="utf-8") as stream:
            combined.extend(csv.DictReader(stream))
    write_csv(RUNS / "evaluation/detection.csv", combined, ["backend", "class", "precision", "recall", "map50", "map50_95"])
    print(f"Detection metrics: {target}")
