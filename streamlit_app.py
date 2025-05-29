# Import the required libraries
import streamlit as st  # For building the web interface
from diffusers import StableDiffusionPipeline  # For generating images with Stable Diffusion
import torch  # For checking GPU availability
from PIL import Image  # For saving and displaying images

# Cache the model so it loads only once and speeds up future generations
@st.cache_resource
def load_model():
    # Check if GPU (cuda) is available; otherwise, use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Choose the model version
    model_id = "runwayml/stable-diffusion-v1-5"

    # Load the model pipeline and move it to the selected device
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    return pipe, device  # Return the model and device info

st.title("Text to Image Generator")

# Take user input (prompt) for generating the image
prompt = st.text_input("Enter a prompt", placeholder="e.g. A magical forest with glowing plants")

if st.button("Generate Image") and prompt:
    # Load the model
    pipe, device = load_model()

    # Show a spinner/loading animation while the image is being generated
    with st.spinner("Generating image..."):
        image = pipe(prompt, num_inference_steps=20, guidance_scale=7.5).images[0]
        image.save("generated_image.png")
        st.image(image, caption="Generated Image", use_container_width=True)
        st.success("Image generated successfully!")
