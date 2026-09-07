from pathlib import Path
import argparse

import cv2


CLASS_NAMES = {
    0: "car",
    1: "bus",
    2: "van",
    3: "others",
}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sequence",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--frame",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    image_path = (
        Path(
            "data/ua_detrac/raw/"
            "DETRAC-Images/DETRAC-Images"
        )
        / args.sequence
        / f"img{args.frame:05d}.jpg"
    )

    label_path = (
        Path("data/ua_detrac/yolo/labels")
        / args.sequence
        / f"img{args.frame:05d}.txt"
    )

    output_path = (
        Path("results/day01")
        / f"{args.sequence}_img{args.frame:05d}_check.jpg"
    )

    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(
            f"Failed to read image: {image_path}"
        )

    if not label_path.exists():
        raise RuntimeError(
            f"Label file does not exist: {label_path}"
        )

    img_h, img_w = image.shape[:2]

    with open(label_path, "r") as f:
        lines = [
            line.strip()
            for line in f.readlines()
            if line.strip()
        ]

    for line in lines:
        values = line.split()

        if len(values) != 5:
            raise ValueError(
                f"Invalid YOLO label row: {line}"
            )

        class_id = int(values[0])

        x_center = float(values[1])
        y_center = float(values[2])
        width = float(values[3])
        height = float(values[4])

        box_w = width * img_w
        box_h = height * img_h

        x1 = int(x_center * img_w - box_w / 2.0)
        y1 = int(y_center * img_h - box_h / 2.0)

        x2 = int(x_center * img_w + box_w / 2.0)
        y2 = int(y_center * img_h + box_h / 2.0)

        class_name = CLASS_NAMES[class_id]

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            image,
            class_name,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(output_path),
        image,
    )

    print("sequence:", args.sequence)
    print("frame:", args.frame)
    print("image_size:", img_w, img_h)
    print("label_count:", len(lines))
    print("saved:", output_path)


if __name__ == "__main__":
    main()
