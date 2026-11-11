# Figurative Language in Robot Discource

`youtube_census_filtered/` is the raw dataset. `dataset/` is built locally and is not committed.

## Build the dataset

From the project root:

```
pip install pandas pyarrow
python scripts/build_dataset.py
```

The script reads `youtube_census_filtered/` and writes:

- `dataset/videos.csv` — one row per video, with hashtags from the title, description, and tags
- `dataset/comments.csv` and `dataset/comments.parquet` — one row per comment or reply, with hashtags from the text
- `dataset/hashtags.csv` — one row per hashtag, from titles, descriptions, tags, and comments

## Transcripts

Run this after `build_dataset.py`. From the project root:

```
pip install youtube-transcript-api yt-dlp faster-whisper imageio-ffmpeg numpy
python scripts/transcripts.py
```

For each video in `dataset/videos.csv`, the script tries English captions in this order: creator-uploaded, YouTube auto-generated, then a Whisper transcription of the downloaded audio. It saves:

- `dataset/transcripts/<video_id>.json` — one file per video
- `dataset/audio/` — audio downloaded for videos that need Whisper
- `dataset/transcripts.csv` — one row per video, with the full transcript text

Finished videos are skipped on a rerun. Useful options:

```
python scripts/transcripts.py --limit 5          # first 5 videos
python scripts/transcripts.py --whisper-only     # skip YouTube captions
python scripts/transcripts.py --model small      # Whisper size: tiny, base, small, medium, large-v3
python scripts/transcripts.py --combine          # rebuild dataset/transcripts.csv only
```
