AI-Powered Text-to-Image Generator

This project is part of my ML Internship assessment for Talrn. The aim was to build a complete text-to-image generation system using open-source models. 
The system takes a text description from the user, processes it through a Stable Diffusion model, and generates images in different styles. 
I built a small web interface so that the application can be used directly in the browser without requiring access to the backend code.

1. Project Overview

The idea behind this project is to understand how modern generative models (especially diffusion models) translate text prompts into images.
My system has four major parts:

User Interface (Streamlit) – where the user types prompts and selects settings
Generator Module – loads the Stable Diffusion model and handles all generation
Utility Layer – watermarking, saving outputs, simple prompt safety checks
Storage – organizes images + metadata in a structured folder

The project focuses on clarity, modularity, and realistic output quality.

2. Architecture

🧱 2. Project Architecture

```bash
ai-image-generator/
├── app.py                 # User interface built with Streamlit
├── generator.py           # Core model logic for text→image
├── utils.py               # Helper functions (watermark, saving, filtering)
├── requirements.txt       # Dependencies
└── outputs/               # Auto-created folders storing images + metadata
```

Flow of the application:

User enters a prompt

Prompt is enriched with style + quality tags

Stable Diffusion generates 1–4 images

Images are watermarked and saved

Results are shown in the UI, with download options


3. Setup & Installation
   
✔️ Step 1: Clone the repository
```bash
git clone https://github.com/<your-username>/ai-image-generator.git
cd ai-image-generator
```

✔️ Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

✔️ Step 3: (Optional) Log in to Hugging Face

If the Stable Diffusion model requires license acceptance:
```bash
pip install huggingface_hub
huggingface-cli login
```

✔️ Step 4: Run the application
```bash
streamlit run app.py
```

Then open the link shown in the terminal (usually):
```bash
http://localhost:8501/
```

🖥️ 4. Hardware Requirements
🔹 GPU (Recommended)

For smooth, fast image generation:

NVIDIA GPU with 8GB VRAM or more

CUDA 11.8 / 12.x installed

Install CUDA-enabled PyTorch:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

🔹 CPU

The system also functions fully on the CPU.

However, image generation will be slower:
~20–40 seconds per image depending on settings

5. How to Use the App

Enter a text description in the prompt box

Select:

Number of images

Style (photorealistic, artistic, cartoon)

Steps

Guidance scale

Negative prompt

Seed

Click Generate

Images appear on the page

Download the PNG / JPEG files

Check metadata inside the expandable section

Prompt is enriched with style + quality tags

Stable Diffusion generates 1–4 images

Images are watermarked and saved

Results are shown in the UI, with download options


6. Technology Stack & Model Details
Technologies Used

Python

Streamlit

PyTorch

HuggingFace Diffusers

Pillow

Accelerate

Model Used
```
runwayml/stable-diffusion-v1-5
```

This is a widely used open-source latent diffusion model trained for general-purpose image generation.

✍️ 7. Prompt Engineering Tips

These tips were applied in the generator to improve image quality:

Add quality terms like:
```
highly detailed, sharp focus, dramatic lighting, 4k resolution
```

Add style-specific hints:
```
photorealistic, digital painting, cartoon style
```

Use context to anchor the scene

Always include a negative prompt, such as:
```
blurry, distorted, low quality, extra limbs, text, watermark
```

The guidance scale of 6–8 works best

Steps between 40–50 usually produce high-quality images

⚠️ 8. Current Limitations

Slower generation on the CPU

Base model outputs are limited to 512×512 resolution

Occasional imperfections depending on the prompt

Only simple prompt filtering (word-based)

Model not fine-tuned on any specific dataset

Needs at least 6–8 GB RAM to run comfortably

🚀 9. Future Improvements

If extended further, the project can include:

Fine-tuning or LoRA training for custom styles

Upgrading to higher-resolution models (e.g., SDXL)

A persistent gallery of past generations

Image-to-Image or Inpainting support

More detailed safety filtering

Prompt templates built into the UI

<img width="800" height="800" alt="image" src="https://github.com/user-attachments/assets/ccfab160-bbca-44d8-b2ec-68e85869adec" />



