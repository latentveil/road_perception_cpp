from pathlib import Path
import shutil


IMAGE_ROOT = Path("data/ua_detrac/raw/DETRAC-Images/DETRAC-Images")
SOURCE_LABEL_ROOT = Path("data/ua_detrac/yolo/labels")
SPLIT_ROOT = Path("configs/splits")
OUTPUT_ROOT = Path("data/ua_detrac/yolo")


def read_sequences(path: Path):
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def create_symlink(source: Path, destination: Path):
    if not source.exists():
        raise RuntimeError(f"Source does not exist: {source}")

    destination.symlink_to(source.resolve())


def prepare_split(split_name: str, sequences):
    image_output_dir = OUTPUT_ROOT / "images" / split_name
    label_output_dir = OUTPUT_ROOT / "labels" / split_name

    if image_output_dir.exists():
        shutil.rmtree(image_output_dir)

    if label_output_dir.exists():
        shutil.rmtree(label_output_dir)

    image_output_dir.mkdir(parents=True, exist_ok=True)
    label_output_dir.mkdir(parents=True, exist_ok=True)

    image_count = 0
    label_count = 0

    for sequence in sequences:
        sequence_image_dir = IMAGE_ROOT / sequence
        sequence_label_dir = SOURCE_LABEL_ROOT / sequence

        label_files = sorted(sequence_label_dir.glob("*.txt"))

        if not label_files:
            raise RuntimeError(f"No labels found for sequence: {sequence}")

        for label_path in label_files:
            frame_stem = label_path.stem
            image_path = sequence_image_dir / f"{frame_stem}.jpg"

            output_stem = f"{sequence}_{frame_stem}"

            output_image_path = image_output_dir / f"{output_stem}.jpg"
            output_label_path = label_output_dir / f"{output_stem}.txt"

            create_symlink(image_path, output_image_path)
            create_symlink(label_path, output_label_path)

            image_count += 1
            label_count += 1

    return image_count, label_count


def main():
    train_sequences = read_sequences(SPLIT_ROOT / "train_sequences.txt")
    val_sequences = read_sequences(SPLIT_ROOT / "val_sequences.txt")

    train_images, train_labels = prepare_split("train", train_sequences)
    val_images, val_labels = prepare_split("val", val_sequences)

    print("train:")
    print("  sequences:", len(train_sequences))
    print("  images:", train_images)
    print("  labels:", train_labels)

    print()
    print("val:")
    print("  sequences:", len(val_sequences))
    print("  images:", val_images)
    print("  labels:", val_labels)

    print()
    print("total_images:", train_images + val_images)
    print("total_labels:", train_labels + val_labels)


if __name__ == "__main__":
    main()