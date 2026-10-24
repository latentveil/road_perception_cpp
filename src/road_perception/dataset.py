"""Prepare detection data from every configured UA-DETRAC sequence."""
import os
import shutil
import xml.etree.ElementTree as ET
from .config import data_path, load_config, require_file, sequences, write_yaml, write_csv

def dataset_yaml(config):
    return require_file(data_path(config["processed"]) / "dataset.yaml")

def prepare(copy_images=False):
    import cv2
    config = load_config("dataset")
    groups = sequences(config)
    annotations, images = data_path(config["annotations"]), data_path(config["images"])
    destination = data_path(config["processed"])
    signature = {"sequences": groups, "names": config["names"], "annotations": config["annotations"], "images": config["images"]}
    manifest = destination / "manifest.yaml"
    if destination.exists() and any(destination.iterdir()):
        import yaml
        if manifest.is_file() and yaml.safe_load(manifest.read_text(encoding="utf-8")) == signature:
            # Recreate the local absolute YOLO path after relocating DATA_ROOT.
            write_yaml(destination / "dataset.yaml", {"path": str(destination), "train": "images/train", "val": "images/val", "names": config["names"]})
            print(f"Prepared dataset already exists: {destination}")
            return
        raise ValueError(f"Processed directory is not empty or incomplete: {destination}. Choose a fresh processed path.")
    for source in (annotations.resolve(), images.resolve()):
        target = destination.resolve()
        if target == source or target.is_relative_to(source) or source.is_relative_to(target):
            raise ValueError("Processed data must be separate from raw data")
    for names in groups.values():
        for name in names:
            require_file(annotations / f"{name}.xml")
            if not (images / name).is_dir():
                raise FileNotFoundError(f"Sequence image directory missing: {images / name}")
    class_ids = {name: int(index) for index, name in config["names"].items()}
    rows = []
    for split, names in groups.items():
        image_out, label_out = destination / "images" / split, destination / "labels" / split
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)
        for name in names:
            root = ET.parse(annotations / f"{name}.xml").getroot()
            if root.attrib.get("name") != name:
                raise ValueError(f"XML sequence name mismatch: {name}")
            frame_count = box_count = 0
            for frame in root.findall("frame"):
                number = int(frame.attrib["num"])
                source = require_file(images / name / f"img{number:05d}.jpg")
                image = cv2.imread(str(source))
                if image is None:
                    raise ValueError(f"Unreadable image: {source}")
                height, width = image.shape[:2]
                stem = f"{name}_img{number:05d}"
                target_image = image_out / f"{stem}.jpg"
                if copy_images:
                    shutil.copy2(source, target_image)
                else:
                    target_image.symlink_to(os.path.relpath(source, target_image.parent))
                labels = []
                for target in frame.findall("target_list/target"):
                    box, attr = target.find("box"), target.find("attribute")
                    if box is None or attr is None:
                        raise ValueError(f"Missing box or attribute: {name}, frame {number}")
                    cls = class_ids[attr.attrib["vehicle_type"]]
                    left, top, w, h = [float(box.attrib[k]) for k in ("left", "top", "width", "height")]
                    x1, y1 = max(0, min(width, left)), max(0, min(height, top))
                    x2, y2 = max(0, min(width, left+w)), max(0, min(height, top+h))
                    if x2 <= x1 or y2 <= y1:
                        raise ValueError(f"Invalid box: {name}, frame {number}")
                    labels.append(f"{cls} {(x1+x2)/2/width:.6f} {(y1+y2)/2/height:.6f} {(x2-x1)/width:.6f} {(y2-y1)/height:.6f}")
                (label_out / f"{stem}.txt").write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
                frame_count += 1
                box_count += len(labels)
            if frame_count == 0:
                raise ValueError(f"No frames in {name}")
            rows.append({"sequence": name, "split": split, "frames": frame_count, "boxes": box_count})
            print(f"{name}: {frame_count} frames, {box_count} boxes", flush=True)
    write_yaml(destination / "dataset.yaml", {"path": str(destination), "train": "images/train", "val": "images/val", "names": config["names"]})
    write_csv(destination / "sequences.csv", rows, ["sequence", "split", "frames", "boxes"])
    write_yaml(manifest, signature)
    print(f"Prepared all {len(rows)} sequences: {destination}")
