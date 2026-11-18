"""Shared video download and bounded-memory FFmpeg frame extraction."""

from pathlib import Path

from .common import CACHE, digest, fingerprint, read_json, write_json
from .download import obtain_video


def prepare_frames(video, video_id, fps, source_hash):
    import imageio_ffmpeg

    reader = imageio_ffmpeg.read_frames(str(video))
    metadata = next(reader)
    reader.close()
    key = fingerprint({"source_sha256": source_hash, "fps": fps, "jpeg_quality": 90,
                       "ffmpeg_version": metadata["ffmpeg_version"], "filter": f"fps={fps}:start_time=0"})
    directory = CACHE / "frames" / video_id / key[:16]
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        frames = [directory / name for name in manifest["files"]]
        if all(p.is_file() and digest(p) == manifest["files"][p.name] for p in frames):
            return frames, metadata
    import numpy as np
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=True)
    reader = imageio_ffmpeg.read_frames(str(video), pix_fmt="rgb24",
                                       output_params=["-vf", f"fps={fps}:start_time=0", "-threads", "2"])
    frames = []
    try:
        metadata = next(reader)
        width, height = metadata["size"]
        for index, raw in enumerate(reader):
            path = directory / f"{index:05d}.jpg"
            temporary = path.with_suffix(".tmp")
            Image.fromarray(np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)).save(
                temporary, format="JPEG", quality=90)
            temporary.replace(path)
            frames.append(path)
    finally:
        reader.close()
    if not frames:
        raise ValueError("FFmpeg produced no frames")
    # Container duration rounding may differ by one sample from FFmpeg's fps filter.
    if abs(len(frames) - metadata["duration"] * fps) > 2:
        raise ValueError("Extracted frame count is inconsistent with the video duration")
    write_json(manifest_path, {"source_sha256": source_hash, "fps": fps, "metadata": metadata,
                              "files": {p.name: digest(p) for p in frames}})
    return frames, metadata
