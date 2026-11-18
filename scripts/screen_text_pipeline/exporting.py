"""Review CSV generation and model-free, atomic multi-video export."""

from .common import EXTRA_COLUMNS, digest, read_csv, read_json, write_csv, write_json
from .review import EXCLUDED, resolve_review

REVIEW_FIELDS = ["event_id", "start", "last_seen", "text", "frame", "suggested_category",
                 "sample_count", "reading_variant_count", "previous_frame", "next_frame",
                 "decision", "corrected_text", "duplicate_of", "merge_into"]


def prepare_review(directory, blocks, frames):
    path = directory / "review.csv"
    if path.exists():
        return
    rows = []
    for block in blocks:
        rows.append({"event_id": block["event_id"], "start": block["start"], "last_seen": block["last_seen"],
                     "text": block["text"], "frame": block["frame"], "suggested_category": block["category"],
                     "sample_count": block["sample_count"], "reading_variant_count": block["reading_variant_count"],
                     "previous_frame": str(frames[max(0, block["first_index"] - 1)].resolve()),
                     "next_frame": str(frames[min(len(frames) - 1, block["last_index"] + 1)].resolve()),
                     "decision": "", "corrected_text": "", "duplicate_of": "", "merge_into": ""})
    write_csv(path, rows, REVIEW_FIELDS)


def reviewed_events(path, blocks):
    fields, rows = read_csv(path)
    if fields != REVIEW_FIELDS:
        raise ValueError(f"Unexpected review columns: {path}")
    indexed = {block["event_id"]: block for block in blocks}
    if len(rows) != len(indexed) or {row["event_id"] for row in rows} != set(indexed):
        raise ValueError(f"Review rows are missing or duplicated: {path}")
    for row in rows:
        block = indexed[row["event_id"]]
        if any(row[key] != str(block[key]) for key in ["start", "last_seen", "text", "frame"]):
            raise ValueError("Original candidate fields changed; use corrected_text for edits")
        if row["decision"].strip().lower() not in EXCLUDED | {"keep", "", "cr_error"}:
            raise ValueError(f"Unknown review decision: {row['decision']}")
        if row["duplicate_of"].strip() and row["merge_into"].strip():
            raise ValueError("Use only one of duplicate_of and merge_into")
        for key in ["duplicate_of", "merge_into"]:
            if row[key].strip() and row[key].strip() not in indexed:
                raise ValueError("Review link points to a missing event")
    if not rows or any(not any(row[key].strip() for key in ["decision", "duplicate_of", "merge_into"]) for row in rows):
        return None
    events, counts, warnings = resolve_review(rows)
    write_json(path.with_name("review_audit.json"), {"review_sha256": digest(path), "events": events,
                                                   "counts": counts, "warnings": warnings})
    return events


def export_summary(output, run_dir, manifest, videos):
    rows = []
    for path in sorted((run_dir / "videos").glob("*/result.json")):
        result = read_json(path)
        video_id = result["video_id"]
        metadata = videos.get(video_id, result["metadata"])
        blocks_path = path.with_name("blocks.json")
        blocks = read_json(blocks_path) if blocks_path.exists() else []
        engine = manifest["config"]["engine"]
        selected = [b for b in blocks if engine == "easyocr" or b["category"] == "overlay"]
        status = result["processing_status"]
        review_status = "draft" if engine == "qwen" else "needs_review"
        if not blocks:
            review_status = "not_reviewed_no_text" if status == "completed" else "unavailable"
        review = path.with_name("review.csv")
        if status == "completed" and review.exists():
            events = reviewed_events(review, blocks)
            if events is not None:
                selected = events
                review_status = "reviewed" if events else "reviewed_no_target_text"
        row = {field: metadata.get(field, "") for field in manifest["metadata_columns"]}
        row.update({"video_url": f"https://www.youtube.com/watch?v={video_id}",
                    "channel_url": f"https://www.youtube.com/channel/{metadata['channel_id']}" if metadata.get("channel_id") else "",
                    "engine": engine, "screen_text": "; ".join(event["text"] for event in selected),
                    "processing_status": status, "review_status": review_status, "error": result.get("error", "")})
        rows.append(row)
    # All reviews are validated before replacing the combined CSV.
    rows.sort(key=lambda row: row["video_id"])
    write_csv(output, rows, manifest["metadata_columns"] + EXTRA_COLUMNS)
    return rows
