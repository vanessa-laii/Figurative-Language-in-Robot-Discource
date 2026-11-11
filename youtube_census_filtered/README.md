# YouTube census data bundle

These data cover YouTube videos about humanoid robots, published 2025-08-01 through 2026-08-31, from searches for `humanoid robot` and `humanoid`, with all comments and replies. The team fetched the videos in late October 2026, and the comments and replies by 2026-11-01. `youtube_census/` contains all collected videos. `youtube_census_filtered/` contains the 278 videos that pass the filters below, with their related data.

## Files

| File | Contents | Full bundle rows | Filtered bundle rows |
|---|---|---:|---:|
| `README.md` | This guide | Not applicable | Not applicable |
| `videos.csv` | Videos | 29,514 | 278 |
| `video_localizations.csv` | Video translations | 169,857 | 2,456 |
| `channels.csv` | Channels | 17,116 | 216 |
| `search_results.csv` | Search results | 33,616 | 449 |
| `ori_screening.csv` | Video classifications and collection status | 29,514 | 278 |
| `comments/short/<YYYY-MM>.csv` | Shorts comments and replies | Part of 7,158,515 combined | Part of 1,231,404 combined |
| `comments/regular/<YYYY-MM>.csv` | Regular video comments and replies | Part of 7,158,515 combined | Part of 1,231,404 combined |

## Join files and select formats

| From | To |
|---|---|
| `videos.id` | `ori_screening.id`, `video_localizations.id` |
| `videos.id` | Comments: `snippet.videoId`, search results: `id.videoId` |
| Videos: `snippet.channelId` | `channels.id` |
| Replies: `snippet.parentId` | Top-level comments: `id` |

Run these pandas examples inside either bundle:

```python
import pandas as pd

videos = pd.read_csv("videos.csv")
screen = pd.read_csv("ori_screening.csv")
videos = videos.merge(screen[["id", "format"]], on="id", validate="one_to_one")
```

Open `comments/short/` for Shorts or `comments/regular/` for regular videos. Regular videos can last under four minutes. For videos, use `format` in `ori_screening.csv`:

```python
shorts = videos.loc[videos["format"].eq("short")]  # use "regular" for other videos
```

In Excel, use XLOOKUP on `id` to add `format` from `ori_screening.csv`, then filter `short` or `regular`.

## Column guide

Dotted names follow YouTube's nested fields. A leading dot below repeats the preceding prefix. List cells contain JSON text. Empty YouTube fields mean YouTube did not send a value, including hidden likes. Empty does not mean zero. Times use UTC. Durations use ISO 8601: `PT1M5S` means 65 seconds.

### videos.csv

| Columns | Meaning |
|---|---|
| `id` | Video ID |
| `snippet.title`, `.description`, `.tags` | Title, description, tags |
| `snippet.publishedAt` | Publication time |
| `snippet.channelId`, `.channelTitle` | Uploader |
| `snippet.defaultAudioLanguage`, `.defaultLanguage`, `.localized.*` | Audio language, text language, localized text |
| `snippet.categoryId`, `topicDetails.*` | Category and topics |
| `statistics.viewCount`, `.likeCount`, `.commentCount`, `.favoriteCount` | Views, likes, comments including replies, favorites |
| `contentDetails.duration` | Duration |

### video_localizations.csv

| Columns | Meaning |
|---|---|
| `id`, `language` | Video ID and translation language |
| `title`, `description` | Translated text |

Videos without translations have no rows here.

### channels.csv

| Columns | Meaning |
|---|---|
| `id` | Channel ID |
| `snippet.title`, `.description`, `.customUrl` | Name, description, address |
| `snippet.publishedAt`, `.country`, `.defaultLanguage` | Creation time, country, language |
| `statistics.subscriberCount`, `.videoCount`, `.viewCount` | Channel totals |
| `statistics.hiddenSubscriberCount` | Hidden subscriber count flag |
| `topicDetails.*` | Topics |

### search_results.csv

Count distinct `id.videoId` values for videos. Repeated videos reflect separate searches. Search text can differ from video details.

| Columns | Meaning |
|---|---|
| `id.videoId` | Video ID |
| `snippet.title`, `.description`, `.channelId`, `.channelTitle` | Search text and uploader |
| `snippet.publishedAt`, `.publishTime` | Video publication time |
| `request.q`, `.week_start` | Search term and weekly window start |
| `request.videoDuration` | Requested length band: `short`, `medium`, `long` |
| `request.page`, `.position` | Page and position, starting at 1 |

### Comment files

Each row represents a top-level comment or reply. Replies reference the top-level comment, including replies to other replies. Only top-level rows contain `thread.*` values.

| Columns | Meaning |
|---|---|
| `id`, `snippet.videoId`, `.channelId` | Comment, video, video owner's channel IDs |
| `snippet.parentId` | Top-level comment ID. Empty for top-level rows. |
| `snippet.textOriginal`, `.textDisplay` | Original text and display text. Display text can contain HTML. |
| `snippet.authorDisplayName`, `.authorChannelId.value` | Author name and channel ID |
| `snippet.likeCount` | Comment likes |
| `snippet.publishedAt`, `.updatedAt` | Publication and update times |
| `thread.id`, `thread.snippet.totalReplyCount` | Thread ID and reported reply count, distinct from saved reply rows |

Other fields across these files: thumbnails, `etag`, `kind`, image URLs, permissions, branding, and other metadata are technical.

### ori_screening.csv

This file holds project classifications and collection status. Other files hold YouTube fields, plus search context in `request.*`. Flags use `1` for yes and `0` for no.

| Column | Meaning |
|---|---|
| `id` | Video ID |
| `format` | `short` or `regular`. Unclear checks count as `short`. |
| `length` | `short`: under 4 minutes, `medium`: 4 to 20 minutes, `long`: over 20. Missing duration uses the search band. |
| `removed_by` | First failed filter: `english`, `robot`, `comments`. Empty means the final set. |
| `comments_fetched_at` | Top-level comment fetch time |
| `comments_status` | `complete`, `commentsDisabled`, or `videoNotFound` |
| `is_english` | English flag |
| `is_short` | Shorts flag. Empty means unclear. |
| `language` | Language code. Empty means detection failed. |
| `language_source` | `declared`, `detected`, or `none` |
| `mentions_robot` | Robot text match flag |
| `month` | Video publication month, `YYYY-MM` |
| `over_1000_comments` | `statistics.commentCount` exceeds 1000. Missing counts fail. |
| `search_duration` | Search length band |
| `shorts_checked_at` | Shorts check time. Empty means no check. |

Filters apply in order:

1. `english`: Use declared audio language except `zxx`, otherwise detect from title and first 500 description characters.
2. `robot`: Match `robot` anywhere in title, description, or tags, ignoring case, including `robotics`.
3. `comments`: Require more than 1000 comments including replies, using `statistics.commentCount` at fetch time.

## Keep in mind

- Counts reflect the fetch date.
- `commentCount` is unreliable for disabled comments. Use `comments_status`.
- Comment filenames span 2025-08 through 2026-08 and follow the video's month, not the comment's month.
- Search returned a limited pool each week, rather than every YouTube video.
- The bundle contains no transcripts.
