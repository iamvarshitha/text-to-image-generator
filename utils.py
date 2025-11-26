"""
utils.py
Helper functions for:
- Prompt safety checking
- Output directory management
- Saving images + JSON metadata
- Adding a simple AI watermark
"""

import os
import json
from pathlib import Path
from typing import Dict

from PIL import Image, ImageDraw, ImageFont

BLOCKED_WORDS = [
    "nude",
    "nudity",
    "nsfw",
    "gore",
    "blood",
    "kill",
    "weapon",
    "gun",
    "sexual",
    "erotic",
]


def is_prompt_allowed(prompt: str) -> bool:

    text = prompt.lower()
    return not any(bad in text for bad in BLOCKED_WORDS)


def ensure_output_dir(base_dir: str, timestamp: str) -> str:

    path = Path(base_dir) / timestamp
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def save_image_with_metadata(
    img: Image.Image,
    metadata: Dict,
    dir_path: str,
    base_name: str = "image",
) -> Dict[str, str]:

    os.makedirs(dir_path, exist_ok=True)

    png_path = os.path.join(dir_path, f"{base_name}.png")
    jpg_path = os.path.join(dir_path, f"{base_name}.jpg")
    json_path = os.path.join(dir_path, f"{base_name}.json")

    img.save(png_path, format="PNG")

    img.convert("RGB").save(jpg_path, format="JPEG", quality=95)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {"png": png_path, "jpg": jpg_path, "json": json_path}


def apply_watermark(
    img: Image.Image,
    text: str = "AI-generated · Talrn Task",
    opacity: int = 150,
    margin: int = 10,
) -> Image.Image:

    if img.mode != "RGBA":
        img = img.convert("RGBA")

    watermark_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(watermark_layer)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = img.width - text_w - margin
    y = img.height - text_h - margin

    draw.text((x, y), text, font=font, fill=(255, 255, 255, opacity))
    combined = Image.alpha_composite(img, watermark_layer)

    return combined.convert("RGB")
