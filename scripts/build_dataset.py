"""Build clean video, comment, and hashtag tables from youtube_census_filtered/.

Outputs (in dataset/):
  videos.csv          one row per video, with hashtags found in title/description/tags
  comments.parquet    one row per comment or reply, with hashtags found in the text
  comments.csv        same as comments.parquet
  hashtags.csv        long format: video_id, comment_id, source, hashtag
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "youtube_census_filtered"
OUT = ROOT / "dataset"
OUT.mkdir(exist_ok=True)

# Not preceded by a word char, "&" (HTML entities), or "/" (URL fragments).
HASHTAG = re.compile(r"(?<![\w&/])#(\w+)", re.UNICODE)


def hashtags(text):
    if not isinstance(text, str):
        return []
    return [h.lower() for h in HASHTAG.findall(text) if not h.isdigit()]


def parse_tags(cell):
    if not isinstance(cell, str) or not cell.strip():
        return []
    try:
        return [t.strip().lower() for t in json.loads(cell)]
    except json.JSONDecodeError:
        return []


videos = pd.read_csv(DATA / "videos.csv")
screen = pd.read_csv(DATA / "ori_screening.csv")
videos = videos.merge(screen[["id", "format", "length", "month"]], on="id", validate="one_to_one")

video_cols = {
    "id": "video_id",
    "format": "format",
    "length": "length",
    "month": "month",
    "snippet.publishedAt": "published_at",
    "snippet.channelId": "channel_id",
    "snippet.channelTitle": "channel_title",
    "snippet.title": "title",
    "snippet.description": "description",
    "snippet.tags": "tags",
    "contentDetails.duration": "duration",
    "contentDetails.caption": "has_captions",
    "statistics.viewCount": "views",
    "statistics.likeCount": "likes",
    "statistics.commentCount": "comment_count",
}
vclean = videos[list(video_cols)].rename(columns=video_cols)
vclean["tags"] = vclean["tags"].map(parse_tags)

tag_rows = []
for r in vclean.itertuples(index=False):
    for h in hashtags(r.title):
        tag_rows.append((r.video_id, None, "title", h))
    for h in hashtags(r.description):
        tag_rows.append((r.video_id, None, "description", h))
    for t in r.tags:
        tag_rows.append((r.video_id, None, "tag", t.lstrip("#")))

vclean["title_hashtags"] = vclean["title"].map(hashtags)
vclean["description_hashtags"] = vclean["description"].map(hashtags)
vclean["all_hashtags"] = [
    sorted(set(a) | set(b)) for a, b in zip(vclean["title_hashtags"], vclean["description_hashtags"])
]
for col in ("tags", "title_hashtags", "description_hashtags", "all_hashtags"):
    vclean[col] = vclean[col].map(json.dumps)
vclean.to_csv(OUT / "videos.csv", index=False, encoding="utf-8")
print(f"videos: {len(vclean)}")

comment_cols = {
    "id": "comment_id",
    "snippet.videoId": "video_id",
    "snippet.parentId": "parent_id",
    "snippet.textOriginal": "text",
    "snippet.authorDisplayName": "author",
    "snippet.authorChannelId.value": "author_channel_id",
    "snippet.likeCount": "likes",
    "snippet.publishedAt": "published_at",
    "snippet.updatedAt": "updated_at",
    "thread.snippet.totalReplyCount": "reply_count",
}
parts = []
for fmt in ("short", "regular"):
    for f in sorted((DATA / "comments" / fmt).glob("*.csv")):
        df = pd.read_csv(f, usecols=lambda c: c in comment_cols, dtype=str, keep_default_na=False)
        df = df.rename(columns=comment_cols)
        df["format"] = fmt
        parts.append(df)
        print(f"  {fmt}/{f.name}: {len(df):,}")
comments = pd.concat(parts, ignore_index=True)
# Many replies arrive without snippet.videoId; inherit it from the top-level comment.
parent_video = comments.loc[comments["parent_id"].eq("")].set_index("comment_id")["video_id"]
missing = comments["video_id"].eq("")
comments.loc[missing, "video_id"] = comments.loc[missing, "parent_id"].map(parent_video).fillna("")
comments = comments[comments["video_id"].isin(vclean["video_id"])]
comments["is_reply"] = comments["parent_id"].ne("")
comments["likes"] = pd.to_numeric(comments["likes"], errors="coerce")
comments["reply_count"] = pd.to_numeric(comments["reply_count"], errors="coerce")
tags_per_comment = comments["text"].map(hashtags)
comments["hashtags"] = tags_per_comment.map(json.dumps)

for cid, vid, tags in zip(comments["comment_id"], comments["video_id"], tags_per_comment):
    for h in tags:
        tag_rows.append((vid, cid, "comment", h))

comments.to_parquet(OUT / "comments.parquet", index=False)
comments.to_csv(OUT / "comments.csv", index=False, encoding="utf-8")
print(f"comments: {len(comments):,} ({comments['is_reply'].sum():,} replies)")

ht = pd.DataFrame(tag_rows, columns=["video_id", "comment_id", "source", "hashtag"])
ht.to_csv(OUT / "hashtags.csv", index=False, encoding="utf-8")
print(f"hashtags: {len(ht):,} rows, {ht['hashtag'].nunique():,} unique")
print(ht.groupby("source")["hashtag"].agg(["count", "nunique"]))
