"""Fetch a transcript for every video in dataset/videos.csv.

Order of preference per video:
  1. YouTube captions in English uploaded by the creator ("manual")
  2. YouTube auto-generated English captions ("auto")
  3. Whisper transcription of the downloaded audio ("whisper")

Resumable: each video is saved to dataset/transcripts/<id>.json and skipped on rerun.
Run scripts/build_dataset.py first. Combine into dataset/transcripts.csv with --combine.

  python scripts/transcripts.py                 # all videos
  python scripts/transcripts.py --limit 5       # quick test
  python scripts/transcripts.py --whisper-only  # skip YouTube captions
  python scripts/transcripts.py --model small   # Whisper size: tiny/base/small/medium/large-v3
  python scripts/transcripts.py --combine
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dataset"
TDIR = OUT / "transcripts"
AUDIO = OUT / "audio"


def from_youtube(video_id):
    from youtube_transcript_api import YouTubeTranscriptApi

    listing = YouTubeTranscriptApi().list(video_id)
    for finder, source in ((listing.find_manually_created_transcript, "manual"),
                           (listing.find_generated_transcript, "auto")):
        try:
            t = finder(["en", "en-US", "en-GB"])
        except Exception:
            continue
        segs = [{"start": s.start, "end": s.start + s.duration, "text": s.text} for s in t.fetch()]
        return source, t.language_code, segs
    return None


def download_audio(video_id):
    import yt_dlp

    AUDIO.mkdir(parents=True, exist_ok=True)
    existing = list(AUDIO.glob(f"{video_id}.*"))
    if existing:
        return existing[0]
    opts = {
        # m4a/webm audio decodes directly in faster-whisper, so ffmpeg is not needed.
        "format": "bestaudio[ext=m4a]/bestaudio",
        "outtmpl": str(AUDIO / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
    return next(AUDIO.glob(f"{video_id}.*"))


def load_audio(path, sr=16000):
    # Decode with the ffmpeg bundled by imageio-ffmpeg; faster-whisper's PyAV decoder breaks on av>=15.
    import subprocess

    import imageio_ffmpeg
    import numpy as np

    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-i", str(path),
           "-f", "s16le", "-ac", "1", "-ar", str(sr), "-loglevel", "error", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


_model = None


def from_whisper(video_id, model_size):
    global _model
    from faster_whisper import WhisperModel

    if _model is None:
        _model = WhisperModel(model_size, device="auto", compute_type="int8")
    audio = load_audio(download_audio(video_id))
    segments, info = _model.transcribe(audio, language="en", vad_filter=True)
    segs = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]
    return f"whisper-{model_size}", info.language, segs


def combine():
    rows = []
    for f in sorted(TDIR.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        rows.append({
            "video_id": d["video_id"],
            "source": d.get("source"),
            "language": d.get("language"),
            "error": d.get("error"),
            "n_segments": len(d.get("segments", [])),
            "transcript": " ".join(s["text"] for s in d.get("segments", [])),
        })
    df = pd.DataFrame(rows)
    titles = pd.read_csv(OUT / "videos.csv").set_index("video_id")["title"]
    df.insert(1, "title", df["video_id"].map(titles))
    df.insert(2, "url", "https://www.youtube.com/watch?v=" + df["video_id"])
    df.to_csv(OUT / "transcripts.csv", index=False, encoding="utf-8")
    print(df["source"].fillna("failed").value_counts())
    print(f"wrote {OUT / 'transcripts.csv'} ({len(df)} videos)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model", default="small")
    ap.add_argument("--whisper-only", action="store_true")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--combine", action="store_true")
    args = ap.parse_args()

    TDIR.mkdir(parents=True, exist_ok=True)
    if args.combine:
        return combine()

    ids = pd.read_csv(OUT / "videos.csv")["video_id"].tolist()[: args.limit]
    for i, vid in enumerate(ids, 1):
        dest = TDIR / f"{vid}.json"
        if dest.exists():
            prev = json.loads(dest.read_text(encoding="utf-8"))
            if not (args.retry_failed and prev.get("error")):
                continue
        record = {"video_id": vid}
        result, errors = None, []
        if not args.whisper_only:
            try:
                result = from_youtube(vid)
            except Exception as e:
                errors.append(f"youtube: {type(e).__name__}")
            time.sleep(1)
        if result is None:
            try:
                result = from_whisper(vid, args.model)
            except Exception as e:
                errors.append(f"whisper: {type(e).__name__}: {e}"[:300])
        if result:
            record["source"], record["language"], record["segments"] = result
        else:
            record["error"] = "; ".join(errors)
        dest.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{i}/{len(ids)}] {vid}: {record.get('source') or record['error']}", flush=True)
    combine()


if __name__ == "__main__":
    main()
