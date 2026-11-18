"""Validate human decisions before exporting reviewed text."""

from collections import Counter
import math
import re

EXCLUDED = {"subtitle", "scene_text", "watermark", "ocr_error"}


def normalize_text(text):
    """Keep internal line breaks, spelling, case, and punctuation."""
    return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")).strip("\n")


def resolve_review(rows):
    """Validate all references before building chronological, reviewed events."""
    indexed, warnings = {}, []
    for position, original in enumerate(rows):
        row = dict(original)
        key = row["event_id"].strip()
        if not re.fullmatch(r"[1-9][0-9]*", key) or key in indexed:
            raise ValueError(f"Invalid or duplicate event_id: {key!r}")
        row["event_id"] = key
        row["decision"] = row.get("decision", "").strip().lower()
        if row["decision"] == "cr_error":
            row["decision"] = "ocr_error"
            warnings.append(f"Event {key}: interpreted cr_error as ocr_error.")
        if row["decision"] not in EXCLUDED | {"keep", ""}:
            raise ValueError(f"Event {key}: unknown decision {row['decision']!r}")
        for name in ["start", "last_seen"]:
            row[name] = float(row[name])
            if not math.isfinite(row[name]) or row[name] < 0:
                raise ValueError(f"Event {key}: invalid {name}")
        if row["last_seen"] < row["start"]:
            raise ValueError(f"Event {key}: last_seen precedes start")
        duplicate = row.get("duplicate_of", "").strip()
        merged = row.get("merge_into", "").strip()
        if duplicate and merged:
            raise ValueError(f"Event {key}: use only one of duplicate_of and merge_into")
        row["target"] = duplicate or merged
        row["relation"] = "duplicate_of" if duplicate else "merge_into" if merged else ""
        row["position"] = position
        if row["target"] and row["decision"] not in {"", "ocr_error"}:
            raise ValueError(f"Event {key}: linked row has conflicting decision")
        if row["target"] and row.get("corrected_text", "").strip():
            raise ValueError(f"Event {key}: place corrected_text on the final keep row")
        if row["target"] and row["decision"]:
            warnings.append(f"Event {key}: explicit {row['relation']} takes precedence over ocr_error.")
        if not row["target"] and not row["decision"]:
            raise ValueError(f"Event {key}: review is incomplete")
        indexed[key] = row

    roots = {}
    for key, row in indexed.items():
        visited, cursor = set(), key
        while indexed[cursor]["target"]:
            if cursor in visited:
                raise ValueError(f"Event {key}: cyclic duplicate/merge references")
            visited.add(cursor)
            cursor = indexed[cursor]["target"]
            if cursor not in indexed:
                raise ValueError(f"Event {key}: missing target {cursor}")
        if row["target"] and indexed[cursor]["decision"] != "keep":
            raise ValueError(f"Event {key}: reference must resolve to a keep row")
        if row["relation"] == "merge_into" and not indexed[cursor].get("corrected_text", "").strip():
            raise ValueError(f"Event {key}: merge destination needs complete corrected_text")
        roots[key] = cursor

    events = []
    for key, row in indexed.items():
        if row["target"] or row["decision"] != "keep":
            continue
        corrected = row.get("corrected_text", "")
        text = normalize_text(corrected if corrected.strip() else row["text"])
        if not text.strip():
            raise ValueError(f"Event {key}: keep text is empty")
        members = [member for member in indexed.values() if roots[member["event_id"]] == key]
        events.append({"event_id": key, "start": min(member["start"] for member in members),
                       "last_seen": max(member["last_seen"] for member in members),
                       "text": text, "source_event_ids": [member["event_id"] for member in members],
                       "frame": row.get("frame", ""), "corrected": bool(corrected.strip()),
                       "position": row["position"]})
    events.sort(key=lambda event: (event["start"], event["position"]))
    for event in events:
        event.pop("position")
    counts = {"review_rows": len(rows), "retained_events": len(events),
              "duplicate_rows": sum(row["relation"] == "duplicate_of" for row in indexed.values()),
              "merged_rows": sum(row["relation"] == "merge_into" for row in indexed.values()),
              "excluded_rows": sum(not row["target"] and row["decision"] in EXCLUDED for row in indexed.values()),
              "corrected_retained_events": sum(event["corrected"] for event in events),
              "multiline_retained_events": sum("\n" in event["text"] for event in events),
              "normalized_decisions": dict(Counter(row["decision"] or "linked" for row in indexed.values()))}
    return events, counts, warnings


