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


