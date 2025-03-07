import streamlit as st
import torch
from diffusers import DiffusionPipeline
from PIL import Image
import io

# Load the model
@st.cache_resource
def load_model():
    model = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", 
                                             torch_dtype=torch.float16, 
                                             use_safetensors=True, 
                                             variant="fp16")
    device = "cpu"
    model.to(device)
    return model

model = load_model()

def generate_image(prompt):
    image = model(prompt).images[0]
    return image

# Streamlit UI
st.title("Text to Image Generator")
st.write("Generate images from text using Stable Diffusion XL")

# User input
prompt = st.text_input("Enter your prompt:", "A beautiful landscape with mountains and a river")
if st.button("Generate Image"):
    with st.spinner("Generating..."):
        image = generate_image(prompt)
        st.image(image, caption="Generated Image", use_column_width=True)
        
        # Allow download
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        st.download_button("Download Image", img_byte_arr.getvalue(), "generated_image.png", "image/png")
