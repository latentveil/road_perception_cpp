import random
from pathlib import Path


SEED = 42
TRAIN_RATIO = 0.8

XML_DIR = Path(
    "data/ua_detrac/raw/Annos/"
    "DETRAC-Train-Annotations-XML/"
    "DETRAC-Train-Annotations-XML"
)

OUTPUT_DIR = Path("configs/splits")
TRAIN_OUTPUT = OUTPUT_DIR / "train_sequences.txt"
VAL_OUTPUT = OUTPUT_DIR / "val_sequences.txt"


def main():
    sequences = sorted(xml_path.stem for xml_path in XML_DIR.glob("*.xml"))

    if len(sequences) != 60:
        raise RuntimeError(
            f"Expected 60 training sequences, but found {len(sequences)}."
        )

    rng = random.Random(SEED)
    rng.shuffle(sequences)

    train_count = int(len(sequences) * TRAIN_RATIO)

    train_sequences = sorted(sequences[:train_count])
    val_sequences = sorted(sequences[train_count:])

    train_set = set(train_sequences)
    val_set = set(val_sequences)

    overlap = train_set & val_set

    if overlap:
        raise RuntimeError(f"Train/val sequence overlap detected: {overlap}")

    if train_set | val_set != set(sequences):
        raise RuntimeError("Some sequences are missing from the split.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    TRAIN_OUTPUT.write_text(
        "\n".join(train_sequences) + "\n",
        encoding="utf-8",
    )

    VAL_OUTPUT.write_text(
        "\n".join(val_sequences) + "\n",
        encoding="utf-8",
    )

    print("seed:", SEED)
    print("total_sequences:", len(sequences))
    print("train_sequences:", len(train_sequences))
    print("val_sequences:", len(val_sequences))
    print("overlap_count:", len(overlap))

    print()
    print("train:")
    for sequence in train_sequences:
        print(" ", sequence)

    print()
    print("val:")
    for sequence in val_sequences:
        print(" ", sequence)

    print()
    print("saved:", TRAIN_OUTPUT)
    print("saved:", VAL_OUTPUT)


if __name__ == "__main__":
    main()