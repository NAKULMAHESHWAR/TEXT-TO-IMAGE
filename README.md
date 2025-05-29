# TEXT-TO-IMAGE

# 🖼️ Text-to-Image Generator with Stable Diffusion

This project uses the [Stable Diffusion](https://huggingface.co/runwayml/stable-diffusion-v1-5) model to generate images based on text prompts. 
It provides both a **Terminal-based interface** and a **web-based interface using Streamlit** for user interaction.

---

## Features

- Generate high-quality images from text descriptions
- Supports both **Terminal** and **Streamlit** web UI
- GPU support (CUDA) for faster image generation
- Saves generated image as `generated_image.png`
- Clean and simple user interface

---

## Technologies Used

- [Python 3.8+](https://www.python.org/)
- [HuggingFace diffusers](https://github.com/huggingface/diffusers)
- [PyTorch](https://pytorch.org/)
- [PIL (Pillow)](https://pillow.readthedocs.io/en/stable/)
- [Streamlit](https://streamlit.io/)

---
pip install -r requirements.txt
You may need to install transformers, torch, diffusers, streamlit, Pillow, etc.

---

## Usage

# Terminal Version
Run the following command:

python app.py
You'll be prompted to enter a description.

The generated image will be saved and displayed.

# Streamlit Web Interface
Start the Streamlit app with:

streamlit run streamlit_app.py
A browser window will open.

Enter your prompt and click "Generate Image" to see the result.

## Example Prompts
"A magical forest with glowing plants"
"A futuristic robot walking in a desert"
"A watercolor painting of a sunset over the mountains"
"Mountains with the river"
"House with Mountains"


