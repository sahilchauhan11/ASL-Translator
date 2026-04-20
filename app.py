import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import pandas as pd
import string

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ASL Alphabet Recognition UI",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM STYLING (The "Amazing UI" part) ---
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Title and headers */
    h1 {
        color: #00E676 !important;
        font-family: 'Inter', sans-serif;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    h2, h3 {
        color: #E0E0E0 !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Prediction Box box */
    .pred-box {
        background: linear-gradient(135deg, #1A1C23 0%, #292D39 100%);
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.4);
        text-align: center;
        border-left: 5px solid #00E676;
    }
    .pred-lbl {
        font-size: 1.2rem;
        color: #A0AAB5;
        margin-bottom: 5px;
    }
    .pred-res {
        font-size: 4rem;
        font-weight: 800;
        color: #FFFFFF;
        text-shadow: 0 0 15px rgba(0, 230, 118, 0.7);
        margin: 0;
    }
    
    /* Progress Bars styling */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #00E676, #00B1FF);
    }
</style>
""", unsafe_allow_html=True)

# --- CACHE MODEL LOADING ---
@st.cache_resource
def load_trained_model():
    """Loads the model locally. Cached so it only loads once."""
    model_path = 'asl_cnn_model.h5'
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)

model = load_trained_model()

# --- DEFINE LABELS ---
# The classes are mapped alphabetically based on folder names in your Kaggle dataset.
# The Ayuraj ASL dataset typically has 29 folders: a-z, del, nothing, space (all lower/upper depending).
# We'll generate an alphabet list. *Note: If your dataset has digits 0-9 as well, add them here!*
LABELS = sorted(list(string.ascii_lowercase)) 
# Add extra labels if they exist in your dataset (e.g., 'del', 'nothing', 'space')
if model is not None and model.output_shape[-1] > 26:
    LABELS.extend(['del', 'nothing', 'space']) 
    LABELS = sorted(LABELS) # Keras flow_from_dataframe sorts alphabetically

# --- SIDEBAR CONTENT ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3233/3233158.png", width=100) # Placeholder Icon
    st.title("About Project")
    st.markdown("""
    **American Sign Language (ASL) Detector**
    
    This AI application classifies images of hands into ASL alphabets. It was built using:
    - **TensorFlow & Keras** (CNN Architecture)
    - **OpenCV & PIL** (Image processing)
    - **Streamlit** (UI/UX)
    
    *Built by YOU!* 
    """)
    st.divider()
    st.write("### Model Status:")
    if model:
        st.success("✅ Neural Network Loaded & Ready!")
    else:
        st.error("❌ Model 'asl_cnn_model.h5' not found. Please upload it to this directory.")

# --- MAIN APP UI ---
st.title("🤟 AI Sign Language Translator")
st.markdown("<h4 style='text-align: center; color: #888;'>Upload a picture of a hand sign and our AI will translate it instantly.</h4>", unsafe_allow_html=True)
st.write("---")

if not model:
    st.warning("⚠️ **Wait!** You need to place your downloaded `asl_cnn_model.h5` inside the exact same folder as this `app.py` script to run the app.")
    st.stop()

# Layout Columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1. Upload Image")
    uploaded_file = st.file_uploader("Upload JPG/PNG of ASL sign", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # Show uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True, channels="RGB")

with col2:
    st.subheader("2. AI Analysis")
    
    if uploaded_file:
        with st.spinner("Analyzing Hand Coordinates & Shapes..."):
            
            # --- IMAGE PREPROCESSING ---
            # 1. Resize to match your CNN's input (64x64)
            img_resized = image.resize((64, 64))
            
            # 2. Convert to Array and normalize (just like rescale=1./255)
            img_array = np.array(img_resized) / 255.0
            
            # 3. Ensure it has 3 channels (RGB)
            if img_array.shape[-1] == 4: # If PNG with alpha
                img_array = img_array[..., :3]
            elif len(img_array.shape) == 2: # If grayscale
                img_array = np.stack((img_array,)*3, axis=-1)
                
            # 4. Expand dimensions to create batch size of 1
            img_array = np.expand_dims(img_array, axis=0)
            
            # --- INFERENCE ---
            predictions = model.predict(img_array)[0] # Get probabilities for all classes
            
            # Find the index of the highest probability
            max_idx = np.argmax(predictions)
            pred_class = LABELS[max_idx] if max_idx < len(LABELS) else f"Class {max_idx}"
            confidence = predictions[max_idx] * 100
            
            # --- DISPLAY RESULTS ---
            st.markdown(f"""
                <div class="pred-box">
                    <div class="pred-lbl">Detected Translation</div>
                    <div class="pred-res">"{str(pred_class).upper()}"</div>
                    <div style="color: #4CAF50; font-weight: bold; margin-top: 10px;">
                        Confidence: {confidence:.1f}%
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.write("---")
            st.write("### Top 3 Probabilities:")
            
            # Get top 3 predictions
            top_3_indices = predictions.argsort()[-3:][::-1]
            for i in top_3_indices:
                class_label = LABELS[i] if i < len(LABELS) else f"Class {i}"
                prob = predictions[i]
                
                # Show label and progress bar
                st.write(f"**{str(class_label).upper()}** - {prob*100:.1f}%")
                st.progress(float(prob))
            
            if confidence > 80:
                st.balloons()
    else:
        # Placeholder when no image is uploaded
        st.info("👈 Upload an image on the left to see the AI magic!")
        # Empty placeholder box to keep UI shape
        st.markdown("""
            <div style="height: 250px; border: 2px dashed #444; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: #666;">
                <i>AI Analysis will appear here...</i>
            </div>
        """, unsafe_allow_html=True)
