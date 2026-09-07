from pathlib import Path

import cv2
import ultralytics
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace, YAML


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models/yolov8s_ua_detrac_best.pt"
VIDEO_PATH = PROJECT_ROOT / "videos/MVI_20012.mp4"
OUTPUT_DIR = PROJECT_ROOT / "results/tracking/explicit_bytetrack"
OUTPUT_PATH = OUTPUT_DIR / "MVI_20012_explicit.mp4"

TRACKER_CONFIG = (
    Path(ultralytics.__file__).resolve().parent
    / "cfg/trackers/bytetrack.yaml"
)

CLASS_NAMES = {
    0: "car",
    1: "bus",
    2: "van",
    3: "others",
}


def main():
    model = YOLO(MODEL_PATH)

    tracker_args = IterableSimpleNamespace(
        **YAML.load(TRACKER_CONFIG)
    )
    tracker = BYTETracker(tracker_args)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {OUTPUT_PATH}")

    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_index += 1

        result = model.predict(
            source=frame,
            imgsz=640,
            conf=0.1,
            iou=0.7,
            device=0,
            verbose=False,
        )[0]

        detections = result.boxes.cpu().numpy()
        tracks = tracker.update(detections, frame)

        for track in tracks:
            x1, y1, x2, y2 = track[:4]
            track_id = int(track[4])
            score = float(track[5])
            class_id = int(track[6])

            x1 = int(round(x1))
            y1 = int(round(y1))
            x2 = int(round(x2))
            y2 = int(round(y2))

            class_name = CLASS_NAMES.get(class_id, str(class_id))
            label = f"{track_id} {class_name} {score:.2f}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        writer.write(frame)

        if frame_index % 100 == 0 or frame_index == frame_count:
            print(
                f"frame {frame_index}/{frame_count}, "
                f"detections={len(detections)}, "
                f"tracks={len(tracks)}"
            )

    cap.release()
    writer.release()

    print()
    print("finished")
    print("frames:", frame_index)
    print("output:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
