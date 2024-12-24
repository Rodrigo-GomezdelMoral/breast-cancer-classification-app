import os
import time
import streamlit as st
from PIL import Image
from utils.new_predictions import load_model, predict_image

st.set_page_config(page_title="Breast Cancer Prediction", layout="centered")

st.title("🔬​ Breast Cancer Prediction App")
st.write("This application uses a Deep Learning model to classify breast cancer images as **Benign** or **Malignant**.")

device, model = load_model()

st.sidebar.title("Available Images")

def list_images(directory):
    """Lists all image files in a directory."""
    return [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

test_dir = "utils/test_data/"
test_images = list_images(test_dir)

st.sidebar.subheader("Test Images")
selected_test_image = st.sidebar.selectbox(
    "Choose a test image:",
    options=test_images,
    format_func=lambda x: f"{x}"
)

user_images_dir = "utils/user_images/"
os.makedirs(user_images_dir, exist_ok=True)
user_images = list_images(user_images_dir)

st.sidebar.subheader("Uploaded Images")
selected_user_image = st.sidebar.selectbox(
    "Choose an uploaded image:",
    options=["None"] + user_images,
    format_func=lambda x: x if x != "None" else "No user image selected"
)

image_path = os.path.join(test_dir, selected_test_image)

if selected_user_image != "None":
    image_path = os.path.join(user_images_dir, selected_user_image)

image = Image.open(image_path).convert("RGB")
st.image(image, caption="Selected Image", use_container_width=True)

if st.button("Predict", help="Click to predict the class of the displayed image"):
    with st.spinner("Predicting... Please wait."):
        time.sleep(1)
        prediction = predict_image(image_path, device, model)
    st.success(f"Prediction: **{prediction}**")

st.subheader("Upload an Image for Prediction")
uploaded_file = st.file_uploader("Select an image in JPG or PNG format", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_path = os.path.join(user_images_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Image uploaded successfully: {uploaded_file.name}")

st.sidebar.info(
    """
    **Instructions**:  
    1. Select an image from the **Test Images** or **Uploaded Images** sections.  
    2. The selected image will appear in the main view.  
    3. Click "Predict" to classify the image.  
    4. Optionally, upload your own images to test and classify.
    """
)

st.markdown("---")
st.markdown(
    """
    **Disclaimer**: This tool is an artificial intelligence-based application for supporting healthcare professionals.  
    It is not a substitute for medical advice, and the healthcare provider's judgment should always prevail.
    """
)
st.markdown("First version v1.0")
