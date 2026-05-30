import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from pixzig_field.dataset import PixZigImageTextDataset, _crop_resize, aspect_bucket, collate_fn


def _write_image(path: Path, size: tuple[int, int] = (24, 32)) -> None:
    image = Image.new("RGB", size, color=(128, 64, 32))
    image.save(path)


def test_sidecar_txt_dataset_requires_matching_caption_by_default(tmp_path):
    _write_image(tmp_path / "sample.png")
    (tmp_path / "sample.txt").write_text("a caption", encoding="utf-8")
    _write_image(tmp_path / "missing.png")
    _write_image(tmp_path / "empty.png")
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")

    dataset = PixZigImageTextDataset(tmp_path, image_size=16, return_path=True)

    assert len(dataset) == 1
    item = dataset[0]
    assert item["image"].shape == (3, 16, 16)
    assert item["text"] == "a caption"
    assert str(item["path"]).endswith("sample.png")


def test_jsonl_dataset_filters_empty_captions_by_default(tmp_path):
    _write_image(tmp_path / "captioned.png")
    _write_image(tmp_path / "empty.png")
    captions = tmp_path / "captions.jsonl"
    captions.write_text(
        "\n".join(
            [
                json.dumps({"image_path": "captioned.png", "text": "a caption"}),
                json.dumps({"image_path": "empty.png", "text": ""}),
            ]
        ),
        encoding="utf-8",
    )

    dataset = PixZigImageTextDataset(tmp_path, image_size=16, captions_file=captions)

    assert len(dataset) == 1
    assert dataset[0]["text"] == "a caption"


def test_jsonl_dataset_strips_captions_and_rejects_invalid_rows(tmp_path):
    _write_image(tmp_path / "captioned.png")
    captions = tmp_path / "captions.jsonl"
    captions.write_text(json.dumps({"image_path": "captioned.png", "text": "  spaced caption  "}), encoding="utf-8")

    dataset = PixZigImageTextDataset(tmp_path, image_size=16, captions_file=captions)

    assert dataset[0]["text"] == "spaced caption"

    bad_captions = tmp_path / "bad.jsonl"
    bad_captions.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        PixZigImageTextDataset(tmp_path, image_size=16, captions_file=bad_captions)

    invalid_captions = tmp_path / "invalid.jsonl"
    invalid_captions.write_text("{bad json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        PixZigImageTextDataset(tmp_path, image_size=16, captions_file=invalid_captions)


def test_csv_dataset_respects_allow_empty_text(tmp_path):
    _write_image(tmp_path / "empty.png")
    captions = tmp_path / "captions.csv"
    captions.write_text("image_path,text\nempty.png,\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no image samples"):
        PixZigImageTextDataset(tmp_path, image_size=16, captions_file=captions)

    dataset = PixZigImageTextDataset(tmp_path, image_size=16, captions_file=captions, allow_empty_text=True)

    assert len(dataset) == 1
    assert dataset[0]["text"] == ""


def test_csv_dataset_strips_captions(tmp_path):
    _write_image(tmp_path / "captioned.png")
    captions = tmp_path / "captions.csv"
    captions.write_text("image_path,text\ncaptioned.png,  spaced caption  \n", encoding="utf-8")

    dataset = PixZigImageTextDataset(tmp_path, image_size=16, captions_file=captions)

    assert dataset[0]["text"] == "spaced caption"


def test_dataset_collate_includes_paths_when_available(tmp_path):
    _write_image(tmp_path / "sample.png")
    (tmp_path / "sample.txt").write_text("caption", encoding="utf-8")
    dataset = PixZigImageTextDataset(tmp_path, image_size=16, crop_mode="random", return_path=True)

    batch = collate_fn([dataset[0]])

    assert batch["image"].shape == (1, 3, 16, 16)
    assert batch["text"] == ["caption"]
    assert len(batch["path"]) == 1
    assert batch["aspect_bucket"] == ["portrait"]
    assert batch["original_size"] == [(32, 24)]
    assert torch.isfinite(batch["image"]).all()


def test_dataset_getitem_bounds_follow_sequence_contract(tmp_path):
    _write_image(tmp_path / "sample.png")
    (tmp_path / "sample.txt").write_text("caption", encoding="utf-8")
    dataset = PixZigImageTextDataset(tmp_path, image_size=16)

    assert dataset[-1]["text"] == "caption"
    with pytest.raises(IndexError, match="out of range"):
        dataset[len(dataset)]
    with pytest.raises(IndexError, match="out of range"):
        dataset[-len(dataset) - 1]


def test_dataset_rejects_invalid_crop_config(tmp_path):
    _write_image(tmp_path / "sample.png")
    (tmp_path / "sample.txt").write_text("caption", encoding="utf-8")

    with pytest.raises(ValueError, match="crop_mode"):
        PixZigImageTextDataset(tmp_path, image_size=16, crop_mode="invalid")

    with pytest.raises(ValueError, match="image_size"):
        PixZigImageTextDataset(tmp_path, image_size=0)

    with pytest.raises(ValueError, match="caption_extension"):
        PixZigImageTextDataset(tmp_path, image_size=16, caption_extension=".")

    with pytest.raises(ValueError, match="horizontal_flip_p"):
        PixZigImageTextDataset(tmp_path, image_size=16, horizontal_flip_p=float("nan"))


def test_center_crop_uses_middle_after_short_side_resize():
    image = Image.fromarray(np.arange(32, dtype=np.uint8)[None, :, None].repeat(16, axis=0).repeat(3, axis=2))

    cropped = _crop_resize(image, image_size=16, crop_mode="center")
    values = np.asarray(cropped)

    assert values.shape == (16, 16, 3)
    assert values[0, 0, 0] == 8
    assert values[0, -1, 0] == 23


def test_random_crop_uses_seeded_random_offset():
    image = Image.fromarray(np.arange(32, dtype=np.uint8)[None, :, None].repeat(16, axis=0).repeat(3, axis=2))
    random.seed(0)

    cropped = _crop_resize(image, image_size=16, crop_mode="random")
    values = np.asarray(cropped)

    assert values.shape == (16, 16, 3)
    assert values[0, 0, 0] == 12
    assert values[0, -1, 0] == 27


def test_pad_crop_resizes_long_side_and_keeps_transparent_borders():
    image = Image.new("RGBA", (32, 16), color=(255, 0, 0, 255))

    cropped = _crop_resize(image, image_size=16, crop_mode="pad")
    values = np.asarray(cropped)

    assert cropped.mode == "RGBA"
    assert values.shape == (16, 16, 4)
    assert values[0, :, 3].max() == 0
    assert values[-1, :, 3].max() == 0
    assert values[4:12, :, 3].min() == 255
    assert values[8, 8, :3].tolist() == [255, 0, 0]


def test_dataset_pad_crop_returns_rgb_training_tensor(tmp_path):
    _write_image(tmp_path / "wide.png", size=(32, 16))
    (tmp_path / "wide.txt").write_text("wide caption", encoding="utf-8")

    dataset = PixZigImageTextDataset(tmp_path, image_size=16, crop_mode="pad")
    item = dataset[0]

    assert item["image"].shape == (3, 16, 16)
    assert item["text"] == "wide caption"
    assert torch.isfinite(item["image"]).all()


def test_aspect_bucket_classification():
    assert aspect_bucket(16, 32) == "portrait"
    assert aspect_bucket(32, 16) == "landscape"
    assert aspect_bucket(24, 24) == "square"
