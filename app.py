# Import the required libraries
from diffusers import StableDiffusionPipeline  # For generating images using Stable Diffusion
import torch  # For using GPU (CUDA) if available
from PIL import Image  # For working with images (saving and displaying)


def main():
    print("Text to Image Generator")
    prompt = input("Enter a prompt to generate an image: ")

    # Check if GPU is available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ID of the pre-trained model to use
    model_id = "runwayml/stable-diffusion-v1-5"
    
    print("Loading model (this may take a few minutes on first run)..")
    
    # Load the Stable Diffusion model pipeline
    # Use float16 for GPU (faster and uses less memory), float32 for CPU
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)  # Move the model to GPU or CPU

    print(" Generating image...")
    image = pipe(prompt, num_inference_steps=20, guidance_scale=7.5).images[0]

    # Save the generated image as a file
    output_path = "generated_image.png"
    image.save(output_path)
    print(f"Image saved as: {output_path}")

    image.show()

if __name__ == "__main__":
    main()
