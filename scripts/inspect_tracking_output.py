from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models/yolov8s_ua_detrac_best.pt"
VIDEO_PATH = PROJECT_ROOT / "videos/MVI_20012.mp4"


def main():
    model = YOLO(MODEL_PATH)

    results = model.track(
        source=VIDEO_PATH,
        tracker="bytetrack.yaml",
        imgsz=640,
        conf=0.25,
        iou=0.7,
        device=0,
        persist=True,
        stream=True,
        verbose=False,
    )

    for frame_index, result in enumerate(results, start=1):
        boxes = result.boxes

        print(f"frame: {frame_index}")
        print(f"boxes.data.shape: {tuple(boxes.data.shape)}")
        print(f"is_track: {boxes.is_track}")

        if boxes.id is not None:
            print("track_id  class  confidence  x1  y1  x2  y2")

            for xyxy, track_id, cls_id, conf in zip(
                boxes.xyxy.cpu(),
                boxes.id.int().cpu(),
                boxes.cls.int().cpu(),
                boxes.conf.cpu(),
            ):
                x1, y1, x2, y2 = xyxy.tolist()

                print(
                    f"{track_id.item():8d}  "
                    f"{cls_id.item():5d}  "
                    f"{conf.item():10.4f}  "
                    f"{x1:6.1f}  {y1:6.1f}  "
                    f"{x2:6.1f}  {y2:6.1f}"
                )

        break


if __name__ == "__main__":
    main()