"""Evaluate the validation split and save per-class detection metrics as CSV."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import argparse
from road_perception.config import cli

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    from road_perception.evaluation import evaluate
    parser.add_argument("--backend", choices=["pt", "onnx"], default="pt")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    cli(lambda: evaluate(backend=args.backend, cpu=args.cpu))
