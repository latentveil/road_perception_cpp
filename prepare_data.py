"""Prepare all configured UA-DETRAC sequences."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import argparse
from road_perception.config import cli

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    from road_perception.dataset import prepare
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of creating symbolic links")
    args = parser.parse_args()
    cli(lambda: prepare(copy_images=args.copy_images))
