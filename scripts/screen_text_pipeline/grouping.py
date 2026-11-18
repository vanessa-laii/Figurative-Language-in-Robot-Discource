"""Temporal consolidation with original readings retained for review."""

from collections import Counter
from difflib import SequenceMatcher


def normalized(text):
    return " ".join(text.split())


def similarity(left, right):
    return SequenceMatcher(None, normalized(left), normalized(right), autojunk=False).ratio()


def overlap(a, b):
    ax0, ax1 = min(p[0] for p in a), max(p[0] for p in a)
    ay0, ay1 = min(p[1] for p in a), max(p[1] for p in a)
    bx0, bx1 = min(p[0] for p in b), max(p[0] for p in b)
    by0, by1 = min(p[1] for p in b), max(p[1] for p in b)
    intersection = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - intersection
    return intersection / union if union else 0


def consolidate(records, engine):
    groups, active = [], []
    seen = set()
    for record in sorted(records, key=lambda item: item["index"]):
        index = record["index"]
        if index in seen:
            raise ValueError("Duplicate frame index")
        seen.add(index)
        if record["status"] != "completed":
            active = []
            continue
        if engine == "qwen":
            categories = sorted({item["category"] for item in record["items"]})
            items = [{"category": category,
                      "text": "\n".join(item["text"] for item in record["items"] if item["category"] == category)}
                     for category in categories]
        else:
            items = record["items"]
        available, current = list(active), []
        for item in items:
            if not item["text"].strip():
                continue
            matches = [g for g in available if g["category"] == item["category"]
                       and g["samples"][-1]["index"] == index - 1
                       and similarity(g["anchor"], item["text"]) >= 0.70
                       and (engine == "qwen" or overlap(g["bbox"], item["bbox"]) >= 0.3)]
            if matches:
                group = max(matches, key=lambda g: similarity(g["anchor"], item["text"]))
                available.remove(group)
            else:
                group = {"category": item["category"], "anchor": item["text"], "samples": []}
                groups.append(group)
            sample = {"index": index, "timestamp": record["timestamp"], "frame": record["frame"],
                      "text": item["text"]}
            group["samples"].append(sample)
            group["bbox"] = item.get("bbox")
            if len(normalized(item["text"])) > len(normalized(group["anchor"])):
                group["anchor"] = item["text"]
            current.append(group)
        active = current
    blocks = []
    for group in groups:
        samples = group["samples"]
        counts = Counter(normalized(s["text"]) for s in samples)
        winner = max(counts, key=lambda text: (counts[text], len(text)))
        matches = [s for s in samples if normalized(s["text"]) == winner]
        representative = matches[len(matches) // 2]
        blocks.append({"event_id": str(len(blocks) + 1), "start": samples[0]["timestamp"],
                       "last_seen": samples[-1]["timestamp"], "text": representative["text"],
                       "frame": representative["frame"], "category": group["category"],
                       "representative_index": representative["index"], "first_index": samples[0]["index"],
                       "last_index": samples[-1]["index"], "sample_count": len(samples),
                       "reading_variant_count": len(counts), "variants": samples})
    return blocks
