"""Generate one input video per configured sequence, including train and val."""
import csv
import math
from .config import load_config, data_path, sequences, write_csv

def make_videos():
    import cv2
    config = load_config("dataset")
    groups = sequences(config)
    names = groups["train"] + groups["val"]
    images, output = data_path(config["images"]), data_path(config["videos"])
    fps = float(config["fps"])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Dataset fps must be positive and finite")
    files = {name: sorted((images / name).glob("*.jpg")) for name in names}
    for name, frames in files.items():
        if not frames:
            raise FileNotFoundError(f"No images for {name}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, name in enumerate(names, 1):
        first = cv2.imread(str(files[name][0]))
        if first is None:
            raise ValueError(f"Unreadable frame: {files[name][0]}")
        height, width = first.shape[:2]
        path = output / f"{name}.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {path}")
        try:
            for source in files[name]:
                frame = cv2.imread(str(source))
                if frame is None or frame.shape[:2] != (height, width):
                    raise ValueError(f"Invalid frame: {source}")
                writer.write(frame)
        finally:
            writer.release()
        capture = cv2.VideoCapture(str(path))
        count = 0
        while capture.read()[0]:
            count += 1
        capture.release()
        if count != len(files[name]):
            raise RuntimeError(f"Encoded frame count mismatch for {name}: {count}")
        manifest.append({"sequence": name, "frames": count, "fps": fps, "width": width, "height": height})
        write_csv(output / "index.csv", manifest, ["sequence", "frames", "fps", "width", "height"])
        print(f"[{index}/{len(names)}] {name}: {count} frames", flush=True)
    print(f"Generated videos for all {len(names)} sequences: {output}")

def required_videos(config):
    groups = sequences(config)
    names = groups["train"] + groups["val"]
    root = data_path(config["videos"])
    if not (root / "index.csv").is_file():
        raise FileNotFoundError("Video index missing. Run python make_videos.py first.")
    with (root / "index.csv").open(newline="", encoding="utf-8") as stream:
        recorded = {row["sequence"] for row in csv.DictReader(stream)}
    if recorded != set(names):
        raise ValueError("Video index does not cover exactly all configured sequences; rerun make_videos.py")
    for name in names:
        if not (root / f"{name}.mp4").is_file():
            raise FileNotFoundError(f"Input video missing: {name}. Run make_videos.py.")
    return [(name, root / f"{name}.mp4") for name in names]
