import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2


CLASS_MAP = {
    "car": 0,
    "bus": 1,
    "van": 2,
    "others": 3,
}


def xywh_abs_to_yolo(
    left: float,
    top: float,
    width: float,
    height: float,
    img_w: int,
    img_h: int,
):
    if img_w <= 0 or img_h <= 0:
        raise ValueError("Image width and height must be positive.")

    if width <= 0 or height <= 0:
        raise ValueError("Bounding-box width and height must be positive.")

    x_center = (left + width / 2.0) / img_w
    y_center = (top + height / 2.0) / img_h
    norm_width = width / img_w
    norm_height = height / img_h

    return x_center, y_center, norm_width, norm_height


def clip_bbox_to_image(
    left: float,
    top: float,
    width: float,
    height: float,
    img_w: int,
    img_h: int,
):
    """
    Clip an absolute (left, top, width, height) bounding box
    to the image boundary.

    Returns:
        clipped_left
        clipped_top
        clipped_width
        clipped_height
        was_clipped
    """
    x1 = left
    y1 = top
    x2 = left + width
    y2 = top + height

    clipped_x1 = max(0.0, min(x1, float(img_w)))
    clipped_y1 = max(0.0, min(y1, float(img_h)))
    clipped_x2 = max(0.0, min(x2, float(img_w)))
    clipped_y2 = max(0.0, min(y2, float(img_h)))

    clipped_width = clipped_x2 - clipped_x1
    clipped_height = clipped_y2 - clipped_y1

    if clipped_width <= 0 or clipped_height <= 0:
        raise ValueError("Bounding box becomes invalid after clipping.")

    was_clipped = (
        clipped_x1 != x1
        or clipped_y1 != y1
        or clipped_x2 != x2
        or clipped_y2 != y2
    )

    return (
        clipped_x1,
        clipped_y1,
        clipped_width,
        clipped_height,
        was_clipped,
    )


def parse_sequence(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    sequence_name = root.attrib["name"]
    frames = []

    for frame in root.findall("frame"):
        frame_num = int(frame.attrib["num"])
        target_list = frame.find("target_list")
        targets = []

        if target_list is not None:
            for target in target_list.findall("target"):
                box = target.find("box")
                attr = target.find("attribute")

                if box is None or attr is None:
                    continue

                vehicle_type = attr.attrib["vehicle_type"]

                if vehicle_type not in CLASS_MAP:
                    raise ValueError(f"Unknown vehicle type: {vehicle_type}")

                targets.append(
                    {
                        "track_id": int(target.attrib["id"]),
                        "vehicle_type": vehicle_type,
                        "left": float(box.attrib["left"]),
                        "top": float(box.attrib["top"]),
                        "width": float(box.attrib["width"]),
                        "height": float(box.attrib["height"]),
                    }
                )

        frames.append(
            {
                "frame_num": frame_num,
                "targets": targets,
            }
        )

    return sequence_name, frames


def convert_sequence(
    xml_path: Path,
    image_root: Path,
    label_root: Path,
):
    sequence_name, frames = parse_sequence(xml_path)

    image_dir = image_root / sequence_name
    output_dir = label_root / sequence_name

    if not image_dir.exists():
        raise RuntimeError(f"Image directory does not exist: {image_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    sequence_box_count = 0
    sequence_clipped_count = 0
    class_counter = Counter()

    for frame in frames:
        frame_num = frame["frame_num"]

        image_path = image_dir / f"img{frame_num:05d}.jpg"
        label_path = output_dir / f"img{frame_num:05d}.txt"

        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        img_h, img_w = image.shape[:2]
        yolo_rows = []

        for target in frame["targets"]:
            vehicle_type = target["vehicle_type"]
            class_id = CLASS_MAP[vehicle_type]

            try:
                (
                    clipped_left,
                    clipped_top,
                    clipped_width,
                    clipped_height,
                    was_clipped,
                ) = clip_bbox_to_image(
                    left=target["left"],
                    top=target["top"],
                    width=target["width"],
                    height=target["height"],
                    img_w=img_w,
                    img_h=img_h,
                )
            except ValueError as error:
                raise RuntimeError(
                    "Invalid bbox after clipping:\n"
                    f"  sequence={sequence_name}\n"
                    f"  frame={frame_num}\n"
                    f"  track_id={target['track_id']}\n"
                    f"  vehicle_type={vehicle_type}\n"
                    f"  bbox=({target['left']}, {target['top']}, "
                    f"{target['width']}, {target['height']})"
                ) from error

            if was_clipped:
                sequence_clipped_count += 1

            xc, yc, w, h = xywh_abs_to_yolo(
                left=clipped_left,
                top=clipped_top,
                width=clipped_width,
                height=clipped_height,
                img_w=img_w,
                img_h=img_h,
            )

            if not (
                0.0 <= xc <= 1.0
                and 0.0 <= yc <= 1.0
                and 0.0 < w <= 1.0
                and 0.0 < h <= 1.0
            ):
                raise RuntimeError(
                    "Invalid normalized bbox:\n"
                    f"  sequence={sequence_name}\n"
                    f"  frame={frame_num}\n"
                    f"  track_id={target['track_id']}\n"
                    f"  normalized_bbox=({xc}, {yc}, {w}, {h})"
                )

            yolo_rows.append(
                f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
            )

            sequence_box_count += 1
            class_counter[vehicle_type] += 1

        with open(label_path, "w") as f:
            for row in yolo_rows:
                f.write(row + "\n")

    return {
        "sequence_name": sequence_name,
        "frame_count": len(frames),
        "box_count": sequence_box_count,
        "clipped_count": sequence_clipped_count,
        "class_counts": class_counter,
    }


def main():
    xml_dir = Path(
        "data/ua_detrac/raw/Annos/"
        "DETRAC-Train-Annotations-XML/"
        "DETRAC-Train-Annotations-XML"
    )

    image_root = Path(
        "data/ua_detrac/raw/DETRAC-Images/DETRAC-Images"
    )

    label_root = Path("data/ua_detrac/yolo/labels")

    xml_files = sorted(xml_dir.glob("*.xml"))

    print("xml_sequence_count:", len(xml_files))

    if len(xml_files) != 60:
        raise RuntimeError(
            f"Expected 60 training XML files, but found {len(xml_files)}."
        )

    total_frames = 0
    total_boxes = 0
    total_clipped_boxes = 0
    total_class_counter = Counter()

    for index, xml_path in enumerate(xml_files, start=1):
        result = convert_sequence(
            xml_path=xml_path,
            image_root=image_root,
            label_root=label_root,
        )

        total_frames += result["frame_count"]
        total_boxes += result["box_count"]
        total_clipped_boxes += result["clipped_count"]
        total_class_counter.update(result["class_counts"])

        print(
            f"[{index:02d}/{len(xml_files)}] "
            f"{result['sequence_name']} "
            f"frames={result['frame_count']} "
            f"boxes={result['box_count']} "
            f"clipped={result['clipped_count']}"
        )

    print()
    print("=" * 60)
    print("Conversion completed.")
    print("sequence_count:", len(xml_files))
    print("total_frames:", total_frames)
    print("total_boxes:", total_boxes)
    print("total_clipped_boxes:", total_clipped_boxes)

    print("class_counts:")
    for class_name in CLASS_MAP:
        print(f"  {class_name}: {total_class_counter[class_name]}")

    print("label_root:", label_root)


if __name__ == "__main__":
    main()
