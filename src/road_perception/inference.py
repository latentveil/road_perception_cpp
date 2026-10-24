"""Process all sequence videos and collect per-sequence and global CSV results."""
from pathlib import Path
import csv
import math
import os
import shutil
import subprocess
import time
from .config import ROOT, RUNS, load_config, project_path, require_file, write_csv, write_yaml
from .video import required_videos

TIME_FIELDS = ["frame", "preprocess_ms", "inference_ms", "postprocess_ms", "tracking_ms", "pipeline_ms", "warmup"]
SUMMARY_FIELDS = ["sequence", "backend", "status", "frames", "unique_tracks", "pipeline_samples", "preprocess_ms", "inference_ms", "postprocess_ms", "tracking_ms", "pipeline_ms", "p50_ms", "p95_ms", "pipeline_fps", "elapsed_seconds", "end_to_end_fps", "error"]

def validate(config, tracker):
    if config["names"] != ["car", "bus", "van", "others"]:
        raise ValueError("The deployment contract requires the four UA-DETRAC classes in their original order")
    if tracker["tracker_type"] != "bytetrack" or not tracker["fuse_score"]:
        raise ValueError("C++ requires ByteTrack and fuse_score=true")
    if abs(float(config["confidence"]) - float(tracker["track_low_thresh"])) > 1e-7:
        raise ValueError("confidence must equal tracker.track_low_thresh")
    if int(config["warmup_frames"]) < 0 or int(config["input_size"]) <= 0:
        raise ValueError("Invalid warmup_frames or input_size")
    for key in ("confidence", "iou"):
        if not math.isfinite(float(config[key])) or not 0 < float(config[key]) < 1:
            raise ValueError(f"Invalid {key}")

def cpp_sequence(name, source, config, tracker, cpu):
    binary = project_path(config["binary"])
    if os.name == "nt" and binary.suffix != ".exe":
        binary = binary.with_suffix(".exe")
    require_file(binary)
    output = RUNS / "videos" / f"{name}.mp4"
    options = {
        "--model": RUNS / "weights/best.onnx", "--source": source, "--output": output,
        "--input-size": config["input_size"], "--device": 0 if cpu else config["device"], "--conf": config["confidence"],
        "--iou": config["iou"], "--warmup": config["warmup_frames"], "--track-high": tracker["track_high_thresh"],
        "--new-track": tracker["new_track_thresh"], "--match": tracker["match_thresh"], "--track-buffer": tracker["track_buffer"],
    }
    command = [str(binary)]
    for key, value in options.items():
        command.extend([key, str(value)])
    if cpu:
        command.append("--cpu")
    env = os.environ.copy()
    if os.name != "nt":
        ort_root = Path(env["ONNXRUNTIME_ROOT"]).expanduser() if env.get("ONNXRUNTIME_ROOT") else project_path(config["onnxruntime"])
        libraries = [str(ort_root / "lib")]
        if env.get("CONDA_PREFIX"):
            libraries.extend(str(p.parent) for p in Path(env["CONDA_PREFIX"]).glob("lib/**/libcudnn.so.9"))
        env["LD_LIBRARY_PATH"] = os.pathsep.join(libraries + [env.get("LD_LIBRARY_PATH", "")])
    result = subprocess.run(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    (RUNS / "timing" / f"{name}.log").write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(result.stdout[-3000:])
    for suffix, folder in ((".tracks.csv", "tracks"), (".timing.csv", "timing")):
        generated = require_file(output.with_suffix(suffix))
        shutil.move(str(generated), str(RUNS / folder / f"{name}.csv"))

def summarize(name, backend, elapsed):
    with (RUNS / "timing" / f"{name}.csv").open(newline="", encoding="utf-8") as stream:
        timings = list(csv.DictReader(stream))
    if not timings:
        raise ValueError(f"No processed frames for {name}")
    with (RUNS / "tracks" / f"{name}.csv").open(newline="", encoding="utf-8") as stream:
        ids = {row["id"] for row in csv.DictReader(stream)}
    selected = [row for row in timings if int(row["warmup"]) == 0]
    row = {"sequence": name, "backend": backend, "status": "ok", "frames": len(timings), "unique_tracks": len(ids),
           "pipeline_samples": len(selected), "elapsed_seconds": elapsed, "end_to_end_fps": len(timings)/elapsed, "error": ""}
    for stage in TIME_FIELDS[1:-1]:
        row[stage] = sum(float(item[stage]) for item in selected)/len(selected) if selected else ""
    values = sorted(float(item["pipeline_ms"]) for item in selected)
    row["p50_ms"] = values[math.ceil(len(values)*.50)-1] if values else ""
    row["p95_ms"] = values[math.ceil(len(values)*.95)-1] if values else ""
    row["pipeline_fps"] = 1000/row["pipeline_ms"] if values and row["pipeline_ms"] > 0 else ""
    return row

def infer(cpu=False):
    config, tracker = load_config("inference"), load_config("tracker")
    validate(config, tracker)
    backend = "cpp"
    dataset = load_config("dataset")
    videos = required_videos(dataset)
    from .config import data_path
    with (data_path(dataset["videos"]) / "index.csv").open(newline="", encoding="utf-8") as stream:
        expected_frames = {row["sequence"]: int(row["frames"]) for row in csv.DictReader(stream)}
    require_file(RUNS / "weights/best.onnx")
    for folder in ("videos", "tracks", "timing"):
        (RUNS / folder).mkdir(parents=True, exist_ok=True)
    binary = project_path(config["binary"])
    require_file(binary.with_suffix(".exe") if os.name == "nt" and binary.suffix != ".exe" else binary)
    write_yaml(RUNS / "inference.yaml", {"inference": config, "tracker": tracker, "backend": backend, "cpu": cpu})
    rows = []
    for index, (name, source) in enumerate(videos, 1):
        print(f"[{index}/{len(videos)}] Processing {name}", flush=True)
        started = time.perf_counter()
        try:
            cpp_sequence(name, source, config, tracker, cpu)
            row = summarize(name, backend, time.perf_counter()-started)
            if row["frames"] != expected_frames[name]:
                raise RuntimeError(f"Frame count mismatch: {row['frames']} processed, {expected_frames[name]} indexed")
        except Exception as error:
            row = {"sequence": name, "backend": backend, "status": "failed", "error": str(error), "elapsed_seconds": time.perf_counter()-started}
            print(f"Failed {name}: {error}", flush=True)
        rows.append(row)
        write_csv(RUNS / "summary.csv", rows, SUMMARY_FIELDS)
    failed = sum(row["status"] != "ok" for row in rows)
    if failed:
        raise RuntimeError(f"{failed}/{len(videos)} sequences failed. See runs/summary.csv; the batch is incomplete.")
    print(f"Completed all {len(videos)} sequences. Summary: {RUNS / 'summary.csv'}")
