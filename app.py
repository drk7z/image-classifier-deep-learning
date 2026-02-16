"""
Streamlit web application for image classification.

Run with: streamlit run app.py
"""

import streamlit as st
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Import from src
import sys
sys.path.insert(0, str(Path(__file__).parent))
from src.predict import ImageClassifier


# Page configuration
st.set_page_config(
    page_title="Image Classifier",
    page_icon="🐱🐶",
    layout="centered"
)

st.title("🐱 vs 🐶 Image Classifier")
st.write("Upload an image to classify it as a cat or dog!")

# Sidebar
st.sidebar.title("Settings")

# Optional uploaded model
uploaded_model_file = st.sidebar.file_uploader(
    "Upload model (.h5)",
    type=["h5"],
    help="Use this option if no trained model is available in the models/ directory."
)

# Model selection
model_name = st.sidebar.selectbox(
    "Select Model",
    ["CNN Model", "Transfer Learning (MobileNetV2)"]
)


def resolve_model_path(default_path, timestamped_pattern):
    """Resolve model path, preferring fixed filename and falling back to latest timestamped file."""
    default_model_path = Path(default_path)

    if default_model_path.exists():
        return str(default_model_path)

    model_candidates = sorted(
        default_model_path.parent.glob(timestamped_pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True
    )

    if model_candidates:
        return str(model_candidates[0])

    return None

# Load model
@st.cache_resource
def load_model(model_path, cache_key=None):
    """Load model from cache."""
    try:
        return ImageClassifier(
            model_path=model_path,
            class_names=['Cat', 'Dog']
        )
    except FileNotFoundError:
        st.error("Model file not found. Please train a model first.")
        return None


# Model paths
model_configs = {
    "CNN Model": {
        "default_path": "models/cnn_classifier_final.h5",
        "timestamped_pattern": "cnn_classifier_final_*.h5"
    },
    "Transfer Learning (MobileNetV2)": {
        "default_path": "models/transfer_learning_final.h5",
        "timestamped_pattern": "transfer_learning_final_*.h5"
    }
}

selected_model_config = model_configs[model_name]
resolved_model_path = None

if uploaded_model_file is not None:
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    uploaded_model_path = models_dir / "uploaded_model.h5"
    uploaded_model_path.write_bytes(uploaded_model_file.getbuffer())
    resolved_model_path = str(uploaded_model_path)
    st.sidebar.success("Uploaded model loaded for this session.")
else:
    resolved_model_path = resolve_model_path(
        selected_model_config["default_path"],
        selected_model_config["timestamped_pattern"]
    )

if resolved_model_path is None:
    st.error("Model file not found. Please train a model first.")
    st.info("You can upload a trained .h5 model from the sidebar to run predictions.")
    st.stop()

model_mtime = Path(resolved_model_path).stat().st_mtime if Path(resolved_model_path).exists() else None
classifier = load_model(resolved_model_path, cache_key=model_mtime)

if classifier is None:
    st.stop()

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
    
    # Make prediction
    with col2:
        st.subheader("Prediction")
        
        # Save temporary file for prediction
        temp_path = Path("temp_image.jpg")
        image.save(temp_path)
        
        # Get prediction
        pred_class, confidence, scores = classifier.predict(
            str(temp_path),
            return_confidence=True
        )
        
        # Display results
        st.metric("Predicted Class", pred_class)
        st.metric("Confidence", f"{confidence:.2%}")
        
        # Display confidence for all classes
        st.subheader("Confidence Scores")
        for i, class_name in enumerate(classifier.class_names):
            st.write(f"{class_name}: {scores[i]:.2%}")
        
        # Visualize confidence
        fig, ax = plt.subplots()
        colors = ['green' if i == np.argmax(scores) else 'lightgray' 
                 for i in range(len(scores))]
        ax.bar(classifier.class_names, scores, color=colors)
        ax.set_ylabel('Confidence')
        ax.set_ylim([0, 1])
        st.pyplot(fig)
        
        # Clean up
        temp_path.unlink()

# Information section
st.sidebar.markdown("---")
st.sidebar.markdown("""
### About
This app uses a CNN trained on the Cats vs Dogs dataset.
The model can classify images with high accuracy.

### Dataset
- Total classes: 2 (Cat, Dog)
- Training approach: Data Augmentation
- Architecture: Custom CNN with 4 convolutional blocks

### Author
Developed as part of image classification project.
""")
