"""
generator.py
A clean, human-friendly wrapper around an open-source Stable Diffusion model.

Responsibilities:
- Load the model on CPU/GPU/MPS
- Apply prompt engineering & negative prompts
- Generate one or more images
- Return images + metadata + saved file paths
"""

import time
from datetime import datetime
from typing import List, Dict, Optional

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

from utils import (
    ensure_output_dir,
    save_image_with_metadata,
    apply_watermark,
    is_prompt_allowed,
)


class Text2ImageGenerator:
    """
    Text2ImageGenerator
    -------------------
    Simple interface to an open-source Stable Diffusion model.

    Example:
        gen = Text2ImageGenerator()
        results = gen.generate_images("a cute robot", num_images=2)
    """

    def __init__(
        self,
        # Use official Stable Diffusion v1.5 as default – very reliable with diffusers
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: Optional[str] = None,
    ):
        """
        model_id:
            Hugging Face model id (must be open-source).
            You can swap this for another compatible diffusers model later.

        device:
            "cuda", "cpu", "mps" or None to auto-detect.
        """

        # Auto-detect best available device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.model_id = model_id

        print(f"[INFO] Loading model '{model_id}' on device: {device}")

        # Use float16 on GPU for speed, float32 on CPU/MPS
        dtype = torch.float16 if device == "cuda" else torch.float32

        try:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                safety_checker=None,  # rely on our own prompt filter + responsible use
            )
        except Exception as e:
            # This is what your Streamlit app will show if download/auth fails
            raise RuntimeError(
                f"Could not load model '{model_id}'. "
                "Make sure you have internet access and have accepted the "
                "model license on Hugging Face (and are logged in via `huggingface-cli login`)."
            ) from e

        self.pipe = self.pipe.to(device)

        if device == "cpu":
            # Slightly reduce memory usage on CPU
            self.pipe.enable_attention_slicing()

        print("[INFO] Model loaded successfully.")

    def _build_enriched_prompt(self, prompt: str, style: str) -> str:
        """
        Add global quality tags + style-specific tags.
        Works well for ANY subject: people, cities, objects, landscapes, etc.
        """
        # Global quality tags
        quality_tags = (
            "highly detailed, sharp focus, dramatic lighting, 4k, "
            "high resolution, well composed, rich colors"
        )

        # Style-specific tags
        style_tags = {
            "photorealistic": (
                "photorealistic, realistic textures, natural light, "
                "professional photography, depth of field, bokeh"
            ),
            "artistic": (
                "digital painting, concept art, highly detailed brush strokes, "
                "cinematic lighting"
            ),
            "cartoon": (
                "cartoon style, clean lines, flat shading, smooth colors, 2d illustration"
            ),
        }

        extra_style = style_tags.get(style.lower(), "")

        enriched = f"{prompt}, {quality_tags}, {extra_style}"
        enriched = enriched.strip().strip(",")  # clean up commas if some parts are empty
        return enriched

    def generate_images(
        self,
        prompt: str,
        num_images: int = 1,
        style: str = "photorealistic",
        guidance_scale: float = 7.0,
        num_inference_steps: int = 40,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        output_dir: str = "outputs",
        add_watermark_flag: bool = True,
    ) -> List[Dict]:
        """
        Generate one or more images from a text prompt.

        Returns:
            List of dicts, each with:
            - "image": PIL.Image
            - "meta": metadata dict
            - "paths": {"png": ..., "jpg": ..., "json": ...}
        """
        if not is_prompt_allowed(prompt):
            raise ValueError("Prompt blocked by content filter. Please use a safer description.")

        enriched_prompt = self._build_enriched_prompt(prompt, style)

        # Strong, general-purpose negative prompt
        if negative_prompt is None:
            negative_prompt = (
                "low quality, blurry, pixelated, grainy, distorted, deformed, extra limbs, "
                "extra fingers, bad anatomy, duplicate objects, text, watermark, logo, "
                "cut off, cropped, out of frame, nsfw, gore, nudity"
            )

        # Optional seed for reproducible generations
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Session folder based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = ensure_output_dir(output_dir, timestamp)

        results: List[Dict] = []

        for idx in range(num_images):
            start = time.time()

            output = self.pipe(
                prompt=enriched_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

            img: Image.Image = output.images[0]

            if add_watermark_flag:
                img = apply_watermark(img)

            elapsed = round(time.time() - start, 2)

            meta = {
                "prompt": prompt,
                "enriched_prompt": enriched_prompt,
                "negative_prompt": negative_prompt,
                "style": style,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "seed": seed,
                "timestamp": timestamp,
                "index": idx,
                "generation_time_sec": elapsed,
                "model_id": self.model_id,
                "device": self.device,
            }

            paths = save_image_with_metadata(img, meta, session_dir, base_name=f"img_{idx:02d}")

            print(f"[INFO] Generated image {idx + 1}/{num_images} in {elapsed} seconds.")

            results.append({"image": img, "meta": meta, "paths": paths})

        return results
