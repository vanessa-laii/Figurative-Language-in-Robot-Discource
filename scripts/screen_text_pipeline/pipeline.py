"""Orchestrate cached downloads, resumable inference, review, and CSV export."""

import argparse
from contextlib import contextmanager
import fcntl
import importlib.metadata
import math
import os
from pathlib import Path
import re
import sys
import time

from .backends import check_device, load_backend, settings
from .common import EXTRA_COLUMNS, ROOT, digest, fingerprint, read_csv, read_json, write_json
from .exporting import export_summary, prepare_review
from .grouping import consolidate
from .media import obtain_video, prepare_frames


def load_videos(path):
    fields, rows = read_csv(path)
    if "video_id" not in fields or set(fields) & set(EXTRA_COLUMNS):
        raise ValueError("Input needs video_id and must not contain reserved output columns")
    videos = {}
    for row in rows:
        key = row["video_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", key) or key in videos:
            raise ValueError(f"Invalid or duplicate video ID: {key}")
        videos[key] = row
    return fields, videos


def make_config(engine, fps, device):
    return {"schema_version": 1, "engine": engine, "fps": fps, "device": device,
            "backend": settings(engine), "similarity": 0.70, "easyocr_iou": 0.3,
            "implementation_sha256": {name: digest(Path(__file__).with_name(name))
                                      for name in ["backends.py", "grouping.py", "media.py", "pipeline.py"]}}


@contextmanager
def lock_run(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another process is using this output directory") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def infer_frames(directory, frames, config, backend_factory, retry_failed=False):
    records = []
    raw_dir = directory / "frames"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(frames):
        dest = raw_dir / f"{index:06d}.json"
        image_hash = digest(path)
        if dest.exists():
            previous = read_json(dest)
            if previous.get("image_sha256") != image_hash or previous.get("timestamp") != index / config["fps"]:
                raise ValueError("Cached frame differs from the current input; use a new output")
            if previous["status"] == "completed" or not retry_failed:
                records.append(previous)
                continue
        backend = backend_factory()
        record = {"index": index, "timestamp": index / config["fps"], "frame": str(path.resolve()),
                  "image_sha256": image_hash, "status": "failed"}
        started = time.perf_counter()
        try:
            record.update(backend.infer(path))
            if not record.get("error"):
                record["status"] = "completed"
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
        record["elapsed_seconds"] = time.perf_counter() - started
        write_json(dest, record)
        records.append(record)
        if (index + 1) % 15 == 0 or index + 1 == len(frames):
            print(f"  Frames {index + 1}/{len(frames)}", flush=True)
    return records


def process_video(video, directory, config, backend_factory, retry_failed=False, cookies=None):
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    previous = read_json(result_path) if result_path.exists() else {}
    result = {**previous, "video_id": video["video_id"], "metadata": video,
              "processing_status": "processing", "error": ""}
    write_json(result_path, result)
    try:
        source = obtain_video(video["video_id"], cookies=cookies) if cookies else obtain_video(video["video_id"])
        source_hash = digest(source)
        if previous.get("source_sha256") and previous["source_sha256"] != source_hash:
            raise ValueError("Source video changed; use a new output")
        result.update({"source": str(source.resolve()), "source_sha256": source_hash})
        write_json(result_path, result)
        frames, metadata = prepare_frames(source, video["video_id"], config["fps"], source_hash)
        result.update({"media": metadata, "frame_count": len(frames)})
        records = infer_frames(directory, frames, config, backend_factory, retry_failed)
        failed = [r["index"] for r in records if r["status"] == "failed"]
        blocks = consolidate(records, config["engine"])
        write_json(directory / "blocks.json", blocks)
        result.update({"processing_status": "partial_failure" if failed else "completed",
                       "failed_frames": failed, "raw_items": sum(len(r.get("items", [])) for r in records),
                       "block_count": len(blocks), "inference_seconds": sum(r["elapsed_seconds"] for r in records),
                       "error": "; ".join(f"Frame {r['index']}: {r['error']}" for r in records if r["status"] == "failed")})
        if not failed:
            prepare_review(directory, blocks, frames)
    except KeyboardInterrupt:
        result.update({"processing_status": "interrupted", "error": "Interrupted; rerun to resume missing frames"})
        write_json(result_path, result)
        raise
    except Exception as error:
        result.update({"processing_status": "failed", "error": f"{type(error).__name__}: {error}"})
    write_json(result_path, result)
    return result


def validate_completed(directory, result):
    """Do not silently skip a run whose source or saved images were changed."""
    if digest(result["source"]) != result["source_sha256"]:
        raise ValueError("Completed run's source video changed; use a new output")
    paths = sorted((directory / "frames").glob("*.json"))
    if len(paths) != result["frame_count"]:
        raise ValueError("Completed run is missing frame records")
    for index, path in enumerate(paths):
        record = read_json(path)
        if (record["index"] != index or record["status"] != "completed"
                or digest(record["frame"]) != record["image_sha256"]):
            raise ValueError("Completed run has changed or incomplete frame records")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Extract English screen text with one local model and export one row per video.")
    parser.add_argument("--engine", choices=["qwen", "easyocr"])
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "dataset/screen_text_results.csv")
    parser.add_argument("--video-id", action="append", help="Repeat this option to select multiple videos")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--device")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--cookies", type=Path, help="Optional local Netscape cookies file for video downloads; not saved in run manifests")
    parser.add_argument("--combine", action="store_true", help="Rebuild CSV from saved results and human decisions; no model is loaded")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        parser.error("--fps must be finite and positive")
    if args.cookies is not None and not args.cookies.expanduser().is_file():
        parser.error("--cookies must point to an existing local Netscape cookies file")
    if args.combine and (args.video_id or args.limit or args.retry_failed or args.cookies):
        parser.error("--combine rebuilds the entire saved run; selection/retry options do not apply")
    output = args.output.resolve()
    run_dir = output.with_suffix(".run")
    manifest_path = run_dir / "manifest.json"
    try:
        if output.suffix.lower() != ".csv":
            raise ValueError("--output must end in .csv")
        if output.exists() and not manifest_path.exists():
            raise ValueError("Existing CSV is not owned by this pipeline; choose a new --output")
        if args.combine and not manifest_path.exists():
            raise ValueError("No saved run exists for this output")
        with lock_run(run_dir):
            manifest = read_json(manifest_path) if manifest_path.exists() else None
            saved = manifest["config"] if manifest else {}
            engine = args.engine or saved.get("engine", "qwen")
            fps = args.fps if args.fps is not None else saved.get("fps", 1.0)
            device = args.device or saved.get("device", "cuda:0")
            input_path = (args.input or Path(manifest["input"] if manifest else ROOT / "dataset/videos.csv")).resolve()
            if output == input_path:
                raise ValueError("Input and output must be different files")
            fields, videos = load_videos(input_path)
            config = saved if args.combine else make_config(engine, fps, device)
            if manifest:
                if (engine != saved["engine"] or fps != saved["fps"] or device != saved["device"]
                        or manifest["config_sha256"] != fingerprint(config)
                        or fields != manifest["metadata_columns"] or str(input_path) != manifest["input"]):
                    raise ValueError("Output is bound to another engine/configuration/input; choose a new --output")
            else:
                manifest = {"config": config, "config_sha256": fingerprint(config), "input": str(input_path),
                            "metadata_columns": fields, "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES")}
                write_json(manifest_path, manifest)
            if args.combine:
                rows = export_summary(output, run_dir, manifest, videos)
                print(f"Combined {len(rows)} videos into {output}")
                return 0
            ids = sorted(set(args.video_id) if args.video_id else videos)
            missing = set(ids) - set(videos)
            if missing:
                raise ValueError(f"Unknown video IDs: {sorted(missing)}")
            ids = ids[:args.limit]
            backend = None

            def backend_factory():
                nonlocal backend
                if backend is None:
                    print(f"Loading {engine} on {device}", flush=True)
                    loaded = load_backend(engine, device)
                    if (run_dir / "model.json").exists() and read_json(run_dir / "model.json") != loaded.metadata:
                        raise ValueError("Model files or device changed; use a new output")
                    backend = loaded
                    write_json(run_dir / "model.json", backend.metadata)
                return backend

            failures = 0
            device_checked = False
            for position, video_id in enumerate(ids, 1):
                directory = run_dir / "videos" / video_id
                previous_path = directory / "result.json"
                previous = read_json(previous_path) if previous_path.exists() else {}
                status = previous.get("processing_status")
                if status == "completed":
                    validate_completed(directory, previous)
                if status == "completed" or (status in {"failed", "partial_failure"} and not args.retry_failed):
                    print(f"[{position}/{len(ids)}] {video_id}: skipped ({status})", flush=True)
                    failures += status in {"failed", "partial_failure"}
                    continue
                if not device_checked:
                    check_device(engine, device)
                    device_checked = True
                print(f"[{position}/{len(ids)}] {video_id}: processing", flush=True)
                try:
                    result = process_video(videos[video_id], directory, config, backend_factory, args.retry_failed, args.cookies)
                except KeyboardInterrupt:
                    export_summary(output, run_dir, manifest, videos)
                    raise
                export_summary(output, run_dir, manifest, videos)
                failures += result["processing_status"] != "completed"
                print(f"  {result['processing_status']}: {result.get('block_count', 0)} blocks {result['error']}", flush=True)
            rows = export_summary(output, run_dir, manifest, videos)
            print(f"Wrote {len(rows)} videos to {output}")
            return 1 if failures else 0
    except (ValueError, OSError, KeyError, importlib.metadata.PackageNotFoundError) as error:
        print(f"Pipeline error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted; completed frames are saved. Rerun the same command to resume.", file=sys.stderr)
        return 130
