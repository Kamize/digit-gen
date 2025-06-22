import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os # For checking if the model file exists

# --- 1. Define the Generator Model Architecture (MUST be identical to training) ---
# Copy the Generator class definition directly from your training script
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + num_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_embedding(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img

# --- 2. Configuration for Model Loading ---
MODEL_PATH = 'generator_model_weights.pth' # Make sure this path is correct
LATENT_DIM = 100
IMG_SIZE = 28
IMG_CHANNELS = 1
NUM_CLASSES = 10
IMG_SHAPE = (IMG_CHANNELS, IMG_SIZE, IMG_SIZE)

# Set device for inference (CPU is fine for Streamlit deployment unless you have a GPU server)
# Streamlit Cloud typically runs on CPU, so it's safer to use 'cpu' here.
# If you were deploying to a service with GPU, you'd use 'cuda'.
device = torch.device("cpu") 

# --- 3. Model Loading Function (with Streamlit caching) ---
@st.cache_resource # Caches the model so it's loaded only once
def load_generator_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Error: Model file not found at {MODEL_PATH}. "
                 "Please ensure 'generator_model_weights.pth' is in the same directory as app.py.")
        st.stop() # Stop the app if model is not found

    model = Generator(LATENT_DIM, NUM_CLASSES, IMG_SHAPE).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # Set to evaluation mode for inference
    return model

# Load the model
generator = load_generator_model()

# --- 4. Function to Generate Images ---
def generate_images(generator_model, digit, num_images, latent_dim, device):
    labels = torch.full((num_images,), digit, dtype=torch.long).to(device)
    noise = torch.randn(num_images, latent_dim).to(device)

    with torch.no_grad():
        generated_images_tensor = generator_model(noise, labels).cpu()

    # Denormalize from [-1, 1] to [0, 1] and convert to PIL Image
    generated_images_tensor = (generated_images_tensor + 1) / 2
    pil_images = [Image.fromarray((img.squeeze().numpy() * 255).astype(np.uint8)) 
                  for img in generated_images_tensor]
    return pil_images

# --- 5. Streamlit App Interface ---
st.set_page_config(page_title="MNIST Digit Generator", layout="centered")

st.title("🔢 MNIST Digit Generator")
st.markdown("Generate 5 unique handwritten digits using a trained Conditional GAN.")

# User selects a digit
selected_digit = st.selectbox(
    "Select a digit to generate:",
    options=list(range(10)),
    index=0 # Default to 0
)

# Button to trigger generation
if st.button(f"Generate 5 images of {selected_digit}"):
    if generator is not None:
        with st.spinner(f"Generating images for digit {selected_digit}..."):
            images = generate_images(generator, selected_digit, 5, LATENT_DIM, device)
        
        st.success("Images generated!")
        st.markdown(f"**Here are 5 generated images for digit {selected_digit}:**")

        # Display images in columns
        cols = st.columns(5)
        for i, img in enumerate(images):
            with cols[i]:
                st.image(img, caption=f"Image {i+1}", use_column_width=True)
    else:
        st.error("Model not loaded. Please check the model file path.")

st.markdown("---")
st.info("This application uses a PyTorch Conditional GAN trained on the MNIST dataset to generate diverse handwritten digits. Generated images are slightly different due to random noise input during generation.")