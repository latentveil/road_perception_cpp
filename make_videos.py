"""Generate input videos for ALL configured train and validation sequences."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import argparse
from road_perception.config import cli

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    from road_perception.video import make_videos
    args = parser.parse_args()
    cli(make_videos)
