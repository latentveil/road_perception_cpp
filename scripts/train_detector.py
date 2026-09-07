from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models/yolov8s.pt"
DATA_PATH = PROJECT_ROOT / "configs/ua_detrac.yaml"
OUTPUT_DIR = PROJECT_ROOT / "results/detection"


def main():
    model = YOLO(MODEL_PATH)

    model.train(
        data=DATA_PATH,
        epochs=50,
        imgsz=640,
        batch=32,
        device=0,
        workers=4,
        project=OUTPUT_DIR,
        name="yolov8s_ua_detrac",
        seed=42,
        deterministic=True,
        patience=15,
        plots=True,
    )


if __name__ == "__main__":
    main()