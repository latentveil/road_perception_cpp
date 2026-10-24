"""Detect and track ALL configured sequence videos; save results and CSV under runs/."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import argparse
from road_perception.config import cli

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    from road_perception.inference import infer
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    cli(lambda: infer(cpu=args.cpu))
