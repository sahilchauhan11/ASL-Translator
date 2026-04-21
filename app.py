import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Load model
model = tf.keras.models.load_model('asl_cnn_model.h5')

# Load class labels
with open('class_indices.json', 'r') as f:
    index_to_class = json.load(f)

IMG_SIZE = 64

st.title("🤟 ASL Sign Language Predictor")

uploaded_file = st.file_uploader("Upload an ASL Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_label = index_to_class[str(predicted_index)]

    st.success(f"Predicted Sign: {predicted_label}")