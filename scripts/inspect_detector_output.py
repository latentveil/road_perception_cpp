from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models/yolov8s_ua_detrac_best.pt"
IMAGE_PATH = PROJECT_ROOT / "data/ua_detrac/yolo/images/val/MVI_20012_img00001.jpg"


def main():
    model = YOLO(MODEL_PATH)

    results = model.predict(
        source=IMAGE_PATH,
        imgsz=640,
        conf=0.25,
        iou=0.7,
        device=0,
        verbose=False,
    )

    boxes = results[0].boxes

    print("boxes.data.shape:", tuple(boxes.data.shape))
    print("is_track:", boxes.is_track)

    if len(boxes) > 0:
        print("first raw detection:")
        print(boxes.data[0].cpu())

        print()
        print("xyxy:", boxes.xyxy[0].cpu())
        print("confidence:", boxes.conf[0].cpu())
        print("class_id:", boxes.cls[0].cpu())
        print("track_id:", boxes.id)


if __name__ == "__main__":
    main()