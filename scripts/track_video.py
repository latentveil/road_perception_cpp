from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models/yolov8s_ua_detrac_best.pt"
VIDEO_PATH = PROJECT_ROOT / "videos/MVI_20012.mp4"
OUTPUT_DIR = PROJECT_ROOT / "results/tracking"


def main():
    model = YOLO(MODEL_PATH)

    model.track(
        source=VIDEO_PATH,
        tracker="bytetrack.yaml",
        imgsz=640,
        conf=0.25,
        iou=0.7,
        device=0,
        persist=True,
        save=True,
        project=OUTPUT_DIR,
        name="bytetrack_baseline",
    )


if __name__ == "__main__":
    main()
