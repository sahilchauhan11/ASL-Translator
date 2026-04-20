# 🤟 AI Sign Language (ASL) Translator

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)
![Keras](https://img.shields.io/badge/Keras-Enabled-red.svg)

## 📌 Overview
This project is an end-to-end Machine Learning web application designed to translate American Sign Language (ASL). Using a custom Convolutional Neural Network (CNN) built with TensorFlow and Keras, the system can classify images of hand signs into the correct English alphabet letter with high accuracy.

The project features a sleek, real-time web interface built with **Streamlit** that allows users to upload images and instantly see the AI's translation, complete with confidence scores and probability graphs.

## 🚀 Features
- **Custom CNN Architecture**: Trained to recognize complex hand shapes from raw pixel data.
- **Image Augmentation**: Utilizes `ImageDataGenerator` for robust training (rotations, zooming, shifting).
- **Interactive Web App**: A beautiful, dark-mode optimized Streamlit UI for immediate inference.
- **Real-Time Confidence Metrics**: Displays the top 3 highest probability predictions visually.

## 🛠️ Technology Stack
- **Deep Learning**: TensorFlow, Keras
- **Web Framework**: Streamlit
- **Data Manipulation**: Pandas, NumPy
- **Image Processing**: PIL (Pillow)
- **Dataset**: Kaggle ASL Dataset

## 🧠 Model Architecture
The underlying model is a Custom Convolutional Neural Network consisting of:
1. Three `Conv2D` layers (32, 64, and 128 filters respectively) with ReLU activation for feature extraction.
2. Three `MaxPooling2D` layers to reduce spatial dimensions and prevent overfitting.
3. A `Flatten` layer followed by a `Dense` fully-connected layer (128 neurons).
4. A `Dropout` layer (0.5) for regularization.
5. A Final `Dense` layer with `softmax` activation to output probabilities across the alphabet classes.

## 💻 How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/YourUsername/asl-translator.git
cd asl-translator
```

**2. Install Dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the Streamlit App**
*(Make sure your trained model `asl_cnn_model.h5` is in the directory!)*
```bash
streamlit run app.py
```

## 📈 Future Improvements
- Integrate live WebCam feed using OpenCV inside the Streamlit app.
- Expand the dataset to include dynamic signs (multi-frame gestures).
- Deploy the model as an API endpoint using FastAPI.

---
*Created as a demonstration of computer vision and end-to-end model deployment.*
