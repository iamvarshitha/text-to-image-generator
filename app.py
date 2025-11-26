"""
app.py
Streamlit web interface for the AI-powered text-to-image generator.

Features:
- Prompt input
- Style selection (photorealistic / artistic / cartoon)
- Adjustable steps, guidance scale, and number of images
- Optional negative prompt + seed
- Progress display
- Image preview + PNG/JPEG downloads
- Metadata viewer
"""

import io
from datetime import datetime

import streamlit as st
from PIL import Image

from generator import Text2ImageGenerator

st.set_page_config(
    page_title="AI Image Generator - Talrn Task",
    layout="wide",
)

st.title("AI-Powered Text-to-Image Generator")
st.caption(
    "Built using open-source Stable Diffusion models. "
    "All images are AI-generated and watermarked."
)

st.sidebar.header("⚙️ Generation Settings")

device_choice = st.sidebar.selectbox(
    "Device",
    options=["auto", "cuda", "cpu"],
    help="Use 'cuda' if you have a compatible GPU. 'auto' will pick the best available.",
)

num_images = st.sidebar.slider(
    "Number of images",
    min_value=1,
    max_value=4,
    value=2,
    help="Generate multiple options for the same prompt.",
)

style = st.sidebar.selectbox(
    "Style guidance",
    options=["photorealistic", "artistic", "cartoon"],
    help="This changes the 'flavour' of the output.",
)

guidance_scale = st.sidebar.slider(
    "Guidance scale",
    min_value=3.0,
    max_value=12.0,
    value=7.0,
    step=0.5,
    help="Higher values = follow text more closely, but too high can look unnatural.",
)

num_inference_steps = st.sidebar.slider(
    "Number of diffusion steps",
    min_value=20,
    max_value=60,
    value=40,
    step=5,
    help="More steps = higher quality, slower generation.",
)

seed_input = st.sidebar.text_input(
    "Random seed (optional)",
    value="",
    help="Use a fixed integer for reproducible outputs. Leave empty for random.",
)

negative_prompt = st.sidebar.text_area(
    "Negative prompt (optional)",
    value=(
        "low quality, blurry, pixelated, grainy, distorted, deformed, extra limbs, "
        "extra fingers, bad anatomy, duplicate objects, text, watermark, logo, "
        "cut off, cropped, out of frame, nsfw, gore, nudity"
    ),
    help="Things you explicitly do NOT want in the image.",
)

add_watermark_flag = st.sidebar.checkbox(
    "Add AI watermark",
    value=True,
    help="Recommended for responsible use.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("💡 Prompt Tips")
st.sidebar.write(
    "- Combine subject + context + style + lighting.\n"
    "- Example: `a futuristic city at night, neon reflections on wet roads, "
    "cinematic lighting, photorealistic`.\n"
    "- Works for people, landscapes, objects, and abstract scenes."
)


prompt = st.text_area(
    "Enter your text prompt",
    placeholder=(
        "Example: a serene mountain lake at sunrise, mist over the water, "
        "ultra detailed landscape, nature photography"
    ),
    height=100,
)

generate_button = st.button("🚀 Generate Images")

if generate_button:
    if not prompt.strip():
        st.error("Please enter a prompt before generating.")
    else:
        with st.spinner("Loading model for the first time (this can take a bit)..."):
            if device_choice == "auto":
                chosen_device = None
            else:
                chosen_device = device_choice

            try:
                generator = Text2ImageGenerator(device=chosen_device)
            except Exception as e:
                st.error(f"Error loading model: {e}")
                st.stop()

        st.success(f"Model is ready on device: {generator.device}")

        # Parse seed
        seed = None
        if seed_input.strip():
            try:
                seed = int(seed_input.strip())
            except ValueError:
                st.warning("Seed must be an integer. Random seed will be used instead.")
                seed = None

        progress = st.progress(0)
        status_text = st.empty()

        try:
            results = generator.generate_images(
                prompt=prompt,
                num_images=num_images,
                style=style,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                negative_prompt=negative_prompt or None,
                seed=seed,
                output_dir="outputs",
                add_watermark_flag=add_watermark_flag,
            )
        except ValueError as ve:
            st.error(str(ve))
            st.stop()
        except Exception as e:
            st.error(f"Error during generation: {e}")
            st.stop()

        cols = st.columns(num_images)

        for idx, data in enumerate(results):
            progress.progress((idx + 1) / num_images)
            status_text.text(
                f"Generated image {idx + 1}/{num_images} "
                f"in {data['meta']['generation_time_sec']} seconds."
            )

            img: Image.Image = data["image"]
            meta = data["meta"]
            paths = data["paths"]  # not used in UI, but good to keep

            with cols[idx]:
                st.image(
                    img,
                    caption=f"Image {idx + 1} · {style}",
                    use_container_width=True,  # fixed deprecation warning
                )

                png_buf = io.BytesIO()
                img.save(png_buf, format="PNG")
                png_buf.seek(0)

                jpg_buf = io.BytesIO()
                img.convert("RGB").save(jpg_buf, format="JPEG", quality=95)
                jpg_buf.seek(0)

                now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name = f"gen_{idx+1}_{now_str}"

                st.download_button(
                    label="⬇️ Download PNG",
                    data=png_buf,
                    file_name=f"{base_name}.png",
                    mime="image/png",
                    key=f"png_{idx}",
                )

                st.download_button(
                    label="⬇️ Download JPEG",
                    data=jpg_buf,
                    file_name=f"{base_name}.jpg",
                    mime="image/jpeg",
                    key=f"jpg_{idx}",
                )

                with st.expander("View metadata"):
                    st.json(meta)

        progress.empty()
        status_text.text("Generation complete.")