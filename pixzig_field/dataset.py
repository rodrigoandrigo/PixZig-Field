from __future__ import annotations

import csv
import json
import math
import random
import warnings
from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def aspect_bucket(width: int, height: int) -> str:
    ratio = width / max(height, 1)
    if ratio < 0.8:
        return "portrait"
    if ratio > 1.25:
        return "landscape"
    return "square"


def _resize_short_side(image: Image.Image, image_size: int) -> Image.Image:
    width, height = image.size
    scale = image_size / min(width, height)
    return image.resize((round(width * scale), round(height * scale)), Image.Resampling.BICUBIC)


def _resize_long_side(image: Image.Image, image_size: int) -> Image.Image:
    width, height = image.size
    scale = image_size / max(width, height)
    resized_width = max(round(width * scale), 1)
    resized_height = max(round(height * scale), 1)
    return image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)


def _pad_resize(image: Image.Image, image_size: int) -> Image.Image:
    resized = _resize_long_side(image.convert("RGBA"), image_size)
    canvas = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
    left = (image_size - resized.width) // 2
    top = (image_size - resized.height) // 2
    canvas.alpha_composite(resized, dest=(left, top))
    return canvas


def _crop_resize(image: Image.Image, image_size: int, crop_mode: str) -> Image.Image:
    if image_size < 1:
        raise ValueError("image_size must be positive")
    if crop_mode == "pad":
        return _pad_resize(image, image_size)
    resized = _resize_short_side(image, image_size)
    if crop_mode == "random":
        max_left = max(resized.width - image_size, 0)
        max_top = max(resized.height - image_size, 0)
        left = random.randint(0, max_left) if max_left > 0 else 0
        top = random.randint(0, max_top) if max_top > 0 else 0
    elif crop_mode == "center":
        left = (resized.width - image_size) // 2
        top = (resized.height - image_size) // 2
    else:
        raise ValueError("crop_mode must be 'center', 'random', or 'pad'")
    return resized.crop((left, top, left + image_size, top + image_size))


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim != 3:
        raise ValueError(f"expected image array with 3 dimensions, got shape {array.shape}")
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor * 2.0 - 1.0


def _clean_caption(value: object) -> str:
    return "" if value is None else str(value).strip()


class PixZigImageTextDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        image_size: int = 512,
        captions_file: str | Path | None = None,
        allow_empty_text: bool = False,
        caption_extension: str = ".txt",
        recursive: bool = True,
        crop_mode: str = "center",
        horizontal_flip_p: float = 0.0,
        return_path: bool = False,
        return_metadata: bool = True,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.allow_empty_text = allow_empty_text
        self.caption_extension = caption_extension if caption_extension.startswith(".") else f".{caption_extension}"
        self.recursive = recursive
        self.crop_mode = crop_mode
        self.horizontal_flip_p = horizontal_flip_p
        self.return_path = return_path
        self.return_metadata = return_metadata
        if image_size < 1:
            raise ValueError("image_size must be positive")
        if self.caption_extension == ".":
            raise ValueError("caption_extension must include a suffix after '.'")
        for name in ("allow_empty_text", "recursive", "return_path", "return_metadata"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        if crop_mode not in {"center", "random", "pad"}:
            raise ValueError("crop_mode must be 'center', 'random', or 'pad'")
        if not math.isfinite(float(horizontal_flip_p)):
            raise ValueError("horizontal_flip_p must be finite")
        if not 0.0 <= horizontal_flip_p <= 1.0:
            raise ValueError("horizontal_flip_p must be in [0, 1]")
        self.samples = self._discover_samples(captions_file)
        if not self.samples:
            raise ValueError(f"no image samples found under {self.root}")

    def _caption_allowed(self, text: str) -> bool:
        return self.allow_empty_text or bool(text.strip())

    def _discover_samples(self, captions_file: str | Path | None) -> list[tuple[Path, str]]:
        if captions_file is not None:
            path = Path(captions_file)
            if not path.is_absolute():
                path = self.root / path
            if path.suffix.lower() == ".jsonl":
                return self._from_jsonl(path)
            if path.suffix.lower() == ".csv":
                return self._from_csv(path)
            raise ValueError(f"unsupported captions file: {path}")
        return self._from_sidecar_txt()

    def _resolve_image(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def _from_jsonl(self, path: Path) -> list[tuple[Path, str]]:
        samples: list[tuple[Path, str]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON in {path} at line {line_number}: {exc.msg}") from exc
                if not isinstance(item, dict):
                    raise ValueError(f"expected JSON object in {path} at line {line_number}")
                image_value = item.get("image_path") or item.get("file_name") or item.get("image")
                text = _clean_caption(item.get("text") if "text" in item else item.get("caption"))
                if image_value is not None and self._caption_allowed(text):
                    samples.append((self._resolve_image(str(image_value)), text))
        return samples

    def _from_csv(self, path: Path) -> list[tuple[Path, str]]:
        samples: list[tuple[Path, str]] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"CSV captions file has no header: {path}")
            for row in reader:
                image_value = row.get("image_path") or row.get("image") or row.get("file_name")
                text = _clean_caption(row.get("text") if "text" in row else row.get("caption"))
                if image_value is not None and self._caption_allowed(text):
                    samples.append((self._resolve_image(image_value), text))
        return samples

    def _from_sidecar_txt(self) -> list[tuple[Path, str]]:
        samples: list[tuple[Path, str]] = []
        paths = self.root.rglob("*") if self.recursive else self.root.glob("*")
        for image_path in sorted(paths):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            caption_path = image_path.with_suffix(self.caption_extension)
            if caption_path.exists():
                text = _clean_caption(caption_path.read_text(encoding="utf-8"))
                if not self._caption_allowed(text):
                    continue
            elif self.allow_empty_text:
                text = ""
            else:
                continue
            samples.append((image_path, text))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_one(self, index: int) -> dict[str, object]:
        image_path, text = self.samples[index]
        with Image.open(image_path) as image:
            image_source = image.convert("RGBA") if self.crop_mode == "pad" else image.convert("RGB")
            original_width, original_height = image_source.size
            cropped = _crop_resize(image_source, self.image_size, self.crop_mode)
            if self.horizontal_flip_p > 0 and random.random() < self.horizontal_flip_p:
                cropped = cropped.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if cropped.mode != "RGB":
                cropped = cropped.convert("RGB")
            tensor = _image_to_tensor(cropped)
        item: dict[str, object] = {"image": tensor, "text": text}
        if self.return_path:
            item["path"] = str(image_path)
        if self.return_metadata:
            item["original_size"] = (original_height, original_width)
            item["aspect_bucket"] = aspect_bucket(original_width, original_height)
        return item

    def __getitem__(self, index: int) -> dict[str, object]:
        if index < 0:
            index += len(self.samples)
        if index < 0 or index >= len(self.samples):
            raise IndexError(f"dataset index out of range: {index}")
        for offset in range(len(self.samples)):
            candidate = (index + offset) % len(self.samples)
            try:
                return self._load_one(candidate)
            except Exception as exc:
                warnings.warn(f"skipping unreadable image {self.samples[candidate][0]}: {exc}")
        raise RuntimeError("all dataset samples failed to load")


def collate_fn(batch: list[dict[str, object]]) -> dict[str, object]:
    output = {
        "image": torch.stack([cast(torch.Tensor, item["image"]) for item in batch]),
        "text": [str(item.get("text", "")) for item in batch],
    }
    if any("path" in item for item in batch):
        output["path"] = [str(item.get("path", "")) for item in batch]
    if any("original_size" in item for item in batch):
        sizes: list[tuple[int, int]] = []
        for item in batch:
            value = item.get("original_size", (0, 0))
            sizes.append(cast(tuple[int, int], value) if isinstance(value, tuple) else (0, 0))
        output["original_size"] = sizes
    if any("aspect_bucket" in item for item in batch):
        output["aspect_bucket"] = [str(item.get("aspect_bucket", "")) for item in batch]
    return output
