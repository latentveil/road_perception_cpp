from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = PROJECT_ROOT / "data/ua_detrac/raw/DETRAC-Images/DETRAC-Images"
VIDEO_ROOT = PROJECT_ROOT / "videos"


def main():
    sequence = "MVI_20012"
    fps = 25.0

    image_dir = IMAGE_ROOT / sequence
    image_paths = sorted(image_dir.glob("*.jpg"))

    if not image_paths:
        raise RuntimeError(f"No images found: {image_dir}")

    first_frame = cv2.imread(str(image_paths[0]))
    if first_frame is None:
        raise RuntimeError(f"Failed to read image: {image_paths[0]}")

    height, width = first_frame.shape[:2]

    VIDEO_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = VIDEO_ROOT / f"{sequence}.mp4"

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    for image_path in image_paths:
        frame = cv2.imread(str(image_path))

        if frame is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        if frame.shape[:2] != (height, width):
            raise RuntimeError(
                f"Unexpected frame size in {image_path}: "
                f"{frame.shape[1]}x{frame.shape[0]}"
            )

        writer.write(frame)

    writer.release()

    print("sequence:", sequence)
    print("frames:", len(image_paths))
    print("resolution:", f"{width}x{height}")
    print("fps:", fps)
    print("output:", output_path)


if __name__ == "__main__":
    main()
