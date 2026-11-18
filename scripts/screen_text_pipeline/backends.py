"""Lazy model adapters: loading one backend never imports the other backend."""

import importlib.metadata
import json
import os

from .common import CACHE, digest, read_json, write_json
from .prompt import PROMPT

MODEL = "Qwen/Qwen3-VL-4B-Instruct"
REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
CATEGORIES = {"overlay", "scene_text", "watermark", "uncertain"}
GENERATION = {"do_sample": False, "max_new_tokens": 768}
PIXELS = {"shortest_edge": 65536, "longest_edge": 1310720}


def parse_response(response):
    text = response.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    value = json.loads(text)
    if not isinstance(value, dict) or not isinstance(value.get("items"), list):
        raise ValueError("Response must contain an items list")
    for item in value["items"]:
        if (not isinstance(item, dict) or not isinstance(item.get("text"), str)
                or not item["text"].strip() or item.get("category") not in CATEGORIES
                or type(item.get("uncertain")) is not bool or not isinstance(item.get("reason"), str)):
            raise ValueError("Invalid Qwen text item")
    return value["items"]


def settings(engine):
    packages = ["torch", "pillow", "imageio-ffmpeg", "numpy"]
    if engine == "qwen":
        packages += ["transformers", "accelerate", "huggingface-hub"]
        config = {"model_id": MODEL, "revision": REVISION, "prompt": PROMPT,
                  "generation": GENERATION, "image_pixel_budget": PIXELS,
                  "dtype": "bfloat16", "attention": "sdpa"}
    else:
        packages += ["easyocr", "torchvision"]
        config = {"detector": "craft", "recognizer": "english_g2", "languages": ["en"],
                  "readtext": {"decoder": "greedy", "batch_size": 8, "canvas_size": 2560,
                               "paragraph": False}, "paragraph_x_ths": 1.0, "paragraph_y_ths": 0.5}
    config["versions"] = {name: importlib.metadata.version(name) for name in packages}
    return config


def check_device(engine, device):
    import torch

    if device == "cpu":
        if engine == "qwen":
            raise ValueError("Qwen requires CUDA in this pipeline; CPU fallback is disabled")
    elif not device.startswith("cuda:") or not torch.cuda.is_available():
        raise ValueError("Requested CUDA device is unavailable; run with GPU access")
    else:
        torch.cuda.get_device_properties(device)
    torch.set_num_threads(4)
    return torch


class QwenBackend:
    def __init__(self, device):
        self.torch = check_device("qwen", device)
        self.device = device
        os.environ.setdefault("HF_HOME", str(CACHE / "hf_cache"))
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        model_dir = CACHE / "models/qwen/model"
        manifest_path = model_dir.parent / "model_manifest.json"
        if not manifest_path.exists():
            from huggingface_hub import snapshot_download

            snapshot_download(MODEL, revision=REVISION, local_dir=str(model_dir),
                              allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
                              max_workers=4)
            write_json(manifest_path, {"model_id": MODEL, "revision": REVISION,
                                      "files": {str(p.relative_to(model_dir)): digest(p)
                                                for p in model_dir.rglob("*")
                                                if p.is_file() and ".cache" not in p.parts}})
        manifest = read_json(manifest_path)
        if manifest["revision"] != REVISION:
            raise ValueError("Cached Qwen revision differs from the pinned model")
        for name, expected in manifest["files"].items():
            if digest(model_dir / name) != expected:
                raise ValueError(f"Qwen model file hash mismatch: {name}")
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
        self.processor.image_processor.size = PIXELS.copy()
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=False, dtype=self.torch.bfloat16,
            device_map={"": device}, attn_implementation="sdpa").eval()
        self.metadata = {"model_directory": str(model_dir), "model_manifest": manifest,
                         "model_device": str(next(self.model.parameters()).device),
                         "gpu": self.torch.cuda.get_device_name(device)}

    def infer(self, path):
        from PIL import Image

        with Image.open(path) as image:
            messages = [{"role": "user", "content": [{"type": "image", "image": image.convert("RGB")},
                                                        {"type": "text", "text": PROMPT}]}]
            inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                        return_dict=True, return_tensors="pt").to(self.device)
            inputs.pop("token_type_ids", None)
            with self.torch.inference_mode():
                generated = self.model.generate(**inputs, **GENERATION)
            text = self.processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:],
                                               skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # Keep invalid raw responses too, so failures can be inspected and retried.
        result = {"raw_response": text, "input_tokens": inputs["input_ids"].shape[1],
                  "output_tokens": generated.shape[1] - inputs["input_ids"].shape[1]}
        try:
            result["items"] = parse_response(text)
        except (ValueError, IndexError) as error:
            result["error"] = f"Invalid model response: {error}"
        return result


def paragraph_text(lines):
    """Keep spatial text rows rather than flattening a title and its support line."""
    rows = []
    for line in sorted(lines, key=lambda item: (min(p[1] for p in item[0]), min(p[0] for p in item[0]))):
        low, high = min(p[1] for p in line[0]), max(p[1] for p in line[0])
        center, height = (low + high) / 2, high - low
        row = next((r for r in rows if abs(center - r["center"]) <= max(height, r["height"]) * 0.5), None)
        if row is None:
            row = {"center": center, "height": height, "lines": []}
            rows.append(row)
        row["lines"].append(line)
    return "\n".join(" ".join(line[1] for line in sorted(row["lines"], key=lambda line: min(p[0] for p in line[0])))
                     for row in rows)


class EasyOCRBackend:
    def __init__(self, device):
        self.torch = check_device("easyocr", device)
        self.device = device
        import easyocr

        model_dir = CACHE / "models/easyocr"
        self.reader = easyocr.Reader(["en"], gpu=False if device == "cpu" else device,
                                     model_storage_directory=str(model_dir),
                                     user_network_directory=str(CACHE / "models/user_network"), verbose=False)
        self.metadata = {"model_directory": str(model_dir),
                         "model_sha256": {p.name: digest(p) for p in model_dir.glob("*.pth")},
                         "model_device": str(next(self.reader.recognizer.parameters()).device)}

    def infer(self, path):
        import numpy as np
        from PIL import Image
        from easyocr.utils import get_paragraph

        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"))
        raw = self.reader.readtext(np.ascontiguousarray(pixels[:, :, ::-1]),
                                   decoder="greedy", batch_size=8, canvas_size=2560, paragraph=False)
        groups = get_paragraph(raw, x_ths=1, y_ths=0.5, mode="ltr")
        items = []
        for box, fallback in sorted(groups, key=lambda group: (min(p[1] for p in group[0]), min(p[0] for p in group[0]))):
            xmin, xmax = min(p[0] for p in box), max(p[0] for p in box)
            ymin, ymax = min(p[1] for p in box), max(p[1] for p in box)
            members = [line for line in raw if xmin <= sum(p[0] for p in line[0]) / 4 <= xmax
                       and ymin <= sum(p[1] for p in line[0]) / 4 <= ymax]
            text = paragraph_text(members) if members else fallback
            if text.strip():
                items.append({"text": text, "bbox": np.asarray(box).tolist(), "category": "candidate",
                              "uncertain": True, "reason": "EasyOCR does not classify text origin"})
        return {"items": items, "raw_lines": [{"bbox": np.asarray(box).tolist(), "text": text,
                                                "confidence": float(confidence)} for box, text, confidence in raw]}


def load_backend(engine, device):
    return QwenBackend(device) if engine == "qwen" else EasyOCRBackend(device)
