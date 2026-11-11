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
