"""Pipeline regression tests using local fixtures and deterministic fake backends."""

from contextlib import ExitStack, redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from screen_text_pipeline import pipeline
from screen_text_pipeline.backends import parse_response, paragraph_text
from screen_text_pipeline.common import read_csv, read_json, write_csv, write_json
from screen_text_pipeline.exporting import REVIEW_FIELDS, reviewed_events
from screen_text_pipeline.grouping import consolidate
from screen_text_pipeline import media
from screen_text_pipeline import download
from screen_text_pipeline.common import digest
from screen_text_pipeline.review import resolve_review


def record(index, text=None, status="completed", box=None):
    item = {"text": text, "category": "candidate" if box else "overlay"}
    if box:
        item["bbox"] = box
    return {"index": index, "timestamp": index / 2, "frame": f"{index}.jpg", "status": status,
            "items": [item] if text else []}


class FakeBackend:
    metadata = {"model": "test"}

    def __init__(self):
        self.calls = []
        self.fail = set()

    def infer(self, path):
        self.calls.append(str(path))
        if str(path) in self.fail:
            raise ValueError("Simulated inference failure")
        return {"items": [{"text": 'Title, "quoted"; original\nSupport line', "category": "overlay",
                           "uncertain": False, "reason": "Visible"}]}


class TextTests(unittest.TestCase):
    def test_json_empty_is_valid_but_invalid_responses_are_not(self):
        self.assertEqual(parse_response('```json\n{"items": []}\n```'), [])
        for text in ['No text', '{"items": [{}]}', '{"items": [{"text": "x", "category": "keep"}]}']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_response(text)

    def test_majority_reading_retains_observed_text_and_lines(self):
        blocks = consolidate([record(0, "Title\nMore than a machine."),
                              record(1, "Title\nMore tha a machine."),
                              record(2, "Title\nMore than a machine.")], "qwen")
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["text"], "Title\nMore than a machine.")
        self.assertEqual(blocks[0]["reading_variant_count"], 2)
        self.assertEqual(blocks[0]["last_seen"], 1)

    def test_empty_missing_and_failed_frames_break_repetitions(self):
        for middle in [[], [record(1)], [record(1, status="failed")]]:
            with self.subTest(middle=middle):
                self.assertEqual(len(consolidate([record(0, "Repeated"), *middle, record(2, "Repeated")], "qwen")), 2)

    def test_progressive_reading_and_distinct_titles(self):
        blocks = consolidate([record(0, "abcd"), record(1, "abcdefg"), record(2, "Different title")], "qwen")
        self.assertEqual([b["text"] for b in blocks], ["abcdefg", "Different title"])

    def test_easyocr_requires_position_overlap(self):
        a = [[0, 0], [10, 0], [10, 10], [0, 10]]
        b = [[20, 20], [30, 20], [30, 30], [20, 30]]
        self.assertEqual(len(consolidate([record(0, "Title", box=a), record(1, "Title", box=a)], "easyocr")), 1)
        self.assertEqual(len(consolidate([record(0, "Title", box=a), record(1, "Title", box=b)], "easyocr")), 2)
        self.assertEqual(paragraph_text([(a, "Title", .9), (b, "Support", .8)]), "Title\nSupport")

    def test_review_forward_merge_and_duplicate_chain(self):
        rows = [{"event_id": str(i), "start": str(i), "last_seen": str(i), "text": "Part",
                 "decision": "", "corrected_text": "", "duplicate_of": "", "merge_into": ""}
                for i in range(1, 4)]
        rows[0]["duplicate_of"] = "2"
        rows[1]["merge_into"] = "3"
        rows[2].update(decision="keep", corrected_text="Full\nParagraph")
        events, _, _ = resolve_review(rows)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["text"], "Full\nParagraph")
        self.assertEqual(events[0]["start"], 1)
        rows[2]["decision"] = ""
        rows[2]["corrected_text"] = ""
        rows[2]["duplicate_of"] = "1"
        with self.assertRaises(ValueError):
            resolve_review(rows)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input = self.root / "videos.csv"
        self.output = self.root / "results.csv"
        self.fields = ["video_id", "title", "channel_id", "description"]
        write_csv(self.input, [{"video_id": key, "title": f"Video {key}", "channel_id": "channel",
                               "description": 'Comma, quote " and\nnewline'} for key in ["a", "b"]], self.fields)
        self.source = self.root / "source.mp4"
        self.source.write_bytes(b"local fixture")
        self.frames = [self.root / f"{i}.jpg" for i in range(2)]
        for frame in self.frames:
            frame.write_bytes(b"frame fixture")
        self.backend = FakeBackend()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(pipeline, "settings", return_value={"model": "fake"}))
        self.device = self.stack.enter_context(patch.object(pipeline, "check_device"))
        self.load = self.stack.enter_context(patch.object(pipeline, "load_backend", return_value=self.backend))
        self.stack.enter_context(patch.object(pipeline, "obtain_video", return_value=self.source))
        self.stack.enter_context(patch.object(pipeline, "prepare_frames", return_value=(self.frames, {"duration": 2})))
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.stack.enter_context(redirect_stderr(io.StringIO()))

    def run_cli(self, *args):
        return pipeline.main(["--input", str(self.input), "--output", str(self.output), *args])

    def test_two_videos_one_csv_model_loaded_once_and_resume_skips(self):
        self.assertEqual(self.run_cli("--engine", "qwen"), 0)
        self.assertEqual(self.load.call_count, 1)
        _, rows = read_csv(self.output)
        self.assertEqual([r["video_id"] for r in rows], ["a", "b"])
        self.assertEqual(rows[0]["screen_text"], 'Title, "quoted"; original\nSupport line')
        self.assertEqual(rows[0]["video_url"], "https://www.youtube.com/watch?v=a")
        self.assertEqual(rows[1]["description"], 'Comma, quote " and\nnewline')
        self.assertEqual(self.run_cli("--engine", "qwen"), 0)
        self.assertEqual(len(self.backend.calls), 4)
        self.assertEqual(self.load.call_count, 1)

    def test_configuration_change_and_unowned_csv_are_protected(self):
        self.assertEqual(self.run_cli("--video-id", "a"), 0)
        original = self.output.read_bytes()
        self.assertEqual(self.run_cli("--fps", "2"), 1)
        self.assertEqual(self.run_cli("--engine", "easyocr"), 1)
        self.assertEqual(self.output.read_bytes(), original)
        other = self.root / "unowned.csv"
        other.write_text("existing")
        self.assertEqual(pipeline.main(["--output", str(other)]), 1)
        self.assertEqual(other.read_text(), "existing")

    def test_frame_failure_retry_only_failed_and_other_video_survives(self):
        self.backend.fail.add(str(self.frames[1]))
        self.assertEqual(self.run_cli(), 1)
        _, rows = read_csv(self.output)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["processing_status"] == "partial_failure" for r in rows))
        self.assertEqual(self.run_cli(), 1)
        self.assertEqual(len(self.backend.calls), 4)
        self.backend.fail.clear()
        self.assertEqual(self.run_cli("--retry-failed"), 0)
        self.assertEqual(len(self.backend.calls), 6)

    def test_download_failure_does_not_abort_other_videos(self):
        with patch.object(pipeline, "obtain_video", side_effect=[ValueError("Unavailable video"), self.source]):
            self.assertEqual(self.run_cli(), 1)
        _, rows = read_csv(self.output)
        self.assertEqual([r["processing_status"] for r in rows], ["failed", "completed"])
        self.assertEqual(rows[0]["screen_text"], "")
        self.assertIn("Unavailable", rows[0]["error"])

    def test_interruption_resumes_missing_frames(self):
        original = self.backend.infer
        with patch.object(self.backend, "infer", side_effect=[original(self.frames[0]), KeyboardInterrupt()]):
            self.assertEqual(self.run_cli("--video-id", "a"), 130)
        self.backend.calls.clear()
        self.assertEqual(self.run_cli("--video-id", "a"), 0)
        self.assertEqual(self.backend.calls, [str(self.frames[1])])

    def test_combine_applies_complete_review_without_loading_model(self):
        self.assertEqual(self.run_cli("--video-id", "a"), 0)
        review = self.output.with_suffix(".run") / "videos/a/review.csv"
        fields, rows = read_csv(review)
        rows[0].update(decision="keep", corrected_text="Corrected\nTitle; punctuation")
        write_csv(review, rows, fields)
        self.load.reset_mock()
        self.device.reset_mock()
        self.assertEqual(self.run_cli("--combine"), 0)
        self.load.assert_not_called()
        self.device.assert_not_called()
        _, result = read_csv(self.output)
        self.assertEqual(result[0]["screen_text"], "Corrected\nTitle; punctuation")
        self.assertEqual(result[0]["review_status"], "reviewed")
        rows[0].update(decision="scene_text", corrected_text="")
        write_csv(review, rows, fields)
        self.assertEqual(self.run_cli("--combine"), 0)
        self.assertEqual(read_csv(self.output)[1][0]["review_status"], "reviewed_no_target_text")

    def test_deleted_review_rows_do_not_overwrite_summary(self):
        self.assertEqual(self.run_cli("--video-id", "a"), 0)
        original = self.output.read_bytes()
        review = self.output.with_suffix(".run") / "videos/a/review.csv"
        write_csv(review, [], REVIEW_FIELDS)
        self.assertEqual(self.run_cli("--combine"), 1)
        self.assertEqual(self.output.read_bytes(), original)

    def test_no_text_is_different_from_failure(self):
        with patch.object(self.backend, "infer", return_value={"items": []}):
            self.assertEqual(self.run_cli("--engine", "easyocr"), 0)
        row = read_csv(self.output)[1][0]
        self.assertEqual(row["processing_status"], "completed")
        self.assertEqual(row["review_status"], "not_reviewed_no_text")
        self.assertEqual(row["error"], "")

    def test_changed_cached_image_is_not_silently_skipped(self):
        self.assertEqual(self.run_cli("--video-id", "a"), 0)
        original = self.output.read_bytes()
        self.frames[0].write_bytes(b"changed")
        self.assertEqual(self.run_cli("--video-id", "a"), 1)
        self.assertEqual(self.output.read_bytes(), original)

    def test_text_blocks_use_semicolon_separator_without_altering_punctuation(self):
        def infer(path):
            text = 'Title, "quoted"; original\nSupport line' if path == self.frames[0] else "Another independent title"
            return {"items": [{"text": text, "category": "overlay"}]}
        with patch.object(self.backend, "infer", side_effect=infer):
            self.assertEqual(self.run_cli("--video-id", "a"), 0)
        text = read_csv(self.output)[1][0]["screen_text"]
        self.assertEqual(text, 'Title, "quoted"; original\nSupport line; Another independent title')


class MediaTests(unittest.TestCase):
    def test_download_options_pass_local_runtime_and_cookies_only_when_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            deno = root / "tools/deno/deno"
            deno.parent.mkdir(parents=True)
            deno.touch()
            cookies = root / "youtube.cookies.txt"
            cookies.write_text("# Netscape HTTP Cookie File\n")
            with patch.object(download, "CACHE", root):
                options = download.download_options(cookies)
                self.assertEqual(options["js_runtimes"], {"deno": {"path": str(deno)}})
                self.assertEqual(options["cookiefile"], str(cookies))
                self.assertNotIn("cookiefile", download.download_options())
                with self.assertRaises(ValueError):
                    download.download_options(root / "missing.txt")

    def test_real_ffmpeg_extracts_complete_sequence_and_reuses_verified_cache(self):
        import imageio_ffmpeg
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "fixture.mp4"
            writer = imageio_ffmpeg.write_frames(str(video), (32, 32), fps=10, codec="libx264")
            writer.send(None)
            for index in range(10):
                writer.send(np.full((32, 32, 3), index * 20, dtype=np.uint8))
            writer.close()
            with patch.object(media, "CACHE", root / "cache"):
                frames, metadata = media.prepare_frames(video, "fixture", 2, digest(video))
                self.assertEqual(len(frames), 2)
                self.assertAlmostEqual(metadata["duration"], 1, places=1)
                before = [p.stat().st_mtime_ns for p in frames]
                repeated, _ = media.prepare_frames(video, "fixture", 2, digest(video))
                self.assertEqual(frames, repeated)
                self.assertEqual(before, [p.stat().st_mtime_ns for p in repeated])
                different, _ = media.prepare_frames(video, "fixture", 1, digest(video))
                self.assertEqual(len(different), 1)
                self.assertNotEqual(frames[0].parent, different[0].parent)


if __name__ == "__main__":
    unittest.main()
