"""Source 500 (text, image) inputs for SemanticGate hand-labeling.

Writes data/gate_data/unlabeled.jsonl + data/gate_data/images/*.jpg.
Each JSONL record has `label: null` — edit by hand to 0, 1, or 2:
    0 = VALID            (grammatical English + identifiable image subject)
    1 = LOW_CONFIDENCE   (ambiguous/truncated text, or blurred/dark image)
    2 = INVALID          (non-English, gibberish, or noise/solid-color image)

The `hint` field is a suggestion based on how the example was constructed.
Trust your own read over the hint — some "likely VALID" Flickr captions are
awkward, and some corrupted examples end up still perfectly parseable.

Requires: `pip install datasets` (not yet pulled in by setup.sh by default).
"""
from __future__ import annotations

import argparse
import json
import random
import string
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


OUT_DIR = Path("data/gate_data")
IMG_DIR = OUT_DIR / "images"
JSONL_PATH = OUT_DIR / "unlabeled.jsonl"

HF_DATASET = "lmms-lab/flickr30k"
HF_SPLIT = "test"

_NON_ENGLISH = [
    "猫は赤い椅子の上で眠っています",
    "Le chat dort sur la chaise rouge en bois",
    "Die Katze schläft auf dem roten Holzstuhl",
    "El gato duerme en la silla roja de madera",
    "Кошка спит на красном деревянном стуле",
    "고양이가 빨간 나무 의자에서 자고 있어요",
    "القطة تنام على الكرسي الخشبي الأحمر",
    "Il gatto dorme sulla sedia rossa di legno",
    "O gato está dormindo na cadeira de madeira vermelha",
    "Kedi kırmızı ahşap sandalyede uyuyor",
    "Mèo đang ngủ trên chiếc ghế gỗ đỏ",
    "แมวกำลังนอนอยู่บนเก้าอี้ไม้สีแดง",
    "חתול ישן על כיסא עץ אדום",
    "Η γάτα κοιμάται στην κόκκινη ξύλινη καρέκλα",
    "गुलाबी कुर्सी पर बिल्ली सो रही है",
]

_TRIVIAL_TEXT = ["", "...", "???", "asdf", "qwertyuiop", "aaaaa", "???!!!", "--", "....."]


def gibberish(min_len: int = 15, max_len: int = 50) -> str:
    length = random.randint(min_len, max_len)
    chars = string.ascii_lowercase + "   "
    return "".join(random.choices(chars, k=length)).strip()


def corrupt_caption(text: str) -> str:
    words = text.split()
    if len(words) < 3:
        return random.choice(["over there", "that one", "the thing", "it's like"])
    style = random.choice(["pronoun", "truncate", "fragment", "stutter"])
    if style == "pronoun":
        n_replace = max(1, len(words) // 3)
        for _ in range(n_replace):
            idx = random.randint(0, len(words) - 1)
            words[idx] = random.choice(["it", "them", "someone", "something", "that"])
        return " ".join(words)
    if style == "truncate":
        k = max(2, len(words) // 3)
        return " ".join(words[:k])
    if style == "fragment":
        return random.choice([
            "the thing over there",
            "you know, that one",
            "it's kind of like",
            "some people maybe",
        ])
    # stutter
    return " ".join(random.choice([w, f"{w} {w}", "uh"]) for w in words[:6])


def heavy_blur(img: Image.Image) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=14))


def darken(img: Image.Image, factor: float = 0.12) -> Image.Image:
    arr = np.asarray(img).astype(np.float32) * factor
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def oversaturate_noise(img: Image.Image) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    arr += np.random.normal(0, 60, arr.shape)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def noise_image(size: int = 224) -> Image.Image:
    arr = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def solid_color(size: int = 224) -> Image.Image:
    color = tuple(random.randint(0, 255) for _ in range(3))
    return Image.new("RGB", (size, size), color)


def checkerboard(size: int = 224, tile: int = 16) -> Image.Image:
    coords = np.indices((size, size)).sum(axis=0)
    arr = ((coords // tile) % 2 * 255).astype(np.uint8)
    return Image.fromarray(np.stack([arr, arr, arr], axis=-1))


def gradient_image(size: int = 224) -> Image.Image:
    x = np.linspace(0, 255, size).astype(np.uint8)
    arr = np.tile(x, (size, 1))
    channel = random.randint(0, 2)
    out = np.zeros((size, size, 3), dtype=np.uint8)
    out[..., channel] = arr
    return Image.fromarray(out)


def stream_flickr(n: int):
    """Yield (caption, PIL image) pairs from Flickr30k. Raises on failure."""
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, split=HF_SPLIT, streaming=True)
    for ex in ds.take(n):
        raw = ex.get("caption") or ex.get("captions") or ex.get("sentences")
        if isinstance(raw, list):
            caption = random.choice(raw)
            if isinstance(caption, dict):
                caption = caption.get("raw") or caption.get("caption") or next(iter(caption.values()))
        else:
            caption = raw
        img = ex["image"].convert("RGB")
        yield caption, img


def build_valid(items, records: list[dict]) -> None:
    for i, (caption, img) in enumerate(items):
        img_path = IMG_DIR / f"valid_{i:03d}.jpg"
        img.save(img_path, quality=85)
        records.append({
            "id": f"valid_{i:03d}",
            "text": caption,
            "image_path": str(img_path),
            "label": None,
            "hint": "likely VALID (0)",
        })


def build_low_conf(items, records: list[dict]) -> None:
    for i, (caption, img) in enumerate(items):
        img_path = IMG_DIR / f"lowconf_{i:03d}.jpg"
        mode = i % 4
        if mode == 0:
            img = heavy_blur(img)
        elif mode == 1:
            img = darken(img)
        elif mode == 2:
            img = oversaturate_noise(img)
        else:
            caption = corrupt_caption(caption)
        if mode in (0, 1, 2) and random.random() < 0.4:
            caption = corrupt_caption(caption)
        img.save(img_path, quality=85)
        records.append({
            "id": f"lowconf_{i:03d}",
            "text": caption,
            "image_path": str(img_path),
            "label": None,
            "hint": "likely LOW_CONFIDENCE (1)",
        })


def build_invalid(n: int, records: list[dict]) -> None:
    for i in range(n):
        img_path = IMG_DIR / f"invalid_{i:03d}.jpg"
        mode = i % 4
        if mode == 0:
            text = gibberish()
            noise_image().save(img_path, quality=85)
        elif mode == 1:
            text = random.choice(_NON_ENGLISH)
            solid_color().save(img_path, quality=85)
        elif mode == 2:
            text = random.choice(_TRIVIAL_TEXT)
            checkerboard().save(img_path, quality=85)
        else:
            text = gibberish() if random.random() < 0.5 else random.choice(_NON_ENGLISH)
            gradient_image().save(img_path, quality=85)
        records.append({
            "id": f"invalid_{i:03d}",
            "text": text,
            "image_path": str(img_path),
            "label": None,
            "hint": "likely INVALID (2)",
        })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-valid", type=int, default=200)
    parser.add_argument("--n-low-conf", type=int, default=150)
    parser.add_argument("--n-invalid", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    IMG_DIR.mkdir(parents=True, exist_ok=True)

    n_flickr = args.n_valid + args.n_low_conf
    print(f"Streaming {n_flickr} Flickr30k records...")
    flickr = list(stream_flickr(n_flickr))
    random.shuffle(flickr)

    valid_items = flickr[: args.n_valid]
    low_conf_items = flickr[args.n_valid : args.n_valid + args.n_low_conf]

    records: list[dict] = []
    print(f"Writing {args.n_valid} valid examples...")
    build_valid(valid_items, records)
    print(f"Writing {args.n_low_conf} low-confidence examples...")
    build_low_conf(low_conf_items, records)
    print(f"Writing {args.n_invalid} invalid examples...")
    build_invalid(args.n_invalid, records)

    random.shuffle(records)

    with JSONL_PATH.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(records)} examples to {JSONL_PATH}")
    print(f"Images in {IMG_DIR}")
    print("\nNext: open the JSONL and set each `label` field to 0, 1, or 2.")


if __name__ == "__main__":
    main()
