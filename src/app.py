import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
import os
from PIL import Image
import io
import json

# Custom theme configuration
st.set_page_config(
    page_title="Blood Group Detection",
    page_icon="🩸",
    layout="wide",
)

# Apply custom CSS for dark theme with green accents
st.markdown("""
    <style>
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        .stButton>button {
            background-color: #00FF00;
            color: black;
        }
        .stProgress .st-bo {
            background-color: #00FF00;
        }
        .success {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: rgba(0, 255, 0, 0.1);
            border: 1px solid #00FF00;
        }
        h1, h2, h3 {
            color: #00FF00 !important;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    model_path = 'models/blood_group_model.h5'
    if not os.path.exists(model_path):
        st.error("Model file not found! Please ensure the model is trained.")
        return None
    return tf.keras.models.load_model(model_path)

def load_model_metrics():
    """Load model metrics from JSON file"""
    metrics_path = 'models/model_metrics.json'
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None

def preprocess_image(image):
    """Preprocess the uploaded image"""
    image = np.array(image)
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if len(image.shape) == 3 and image.shape[-1] == 3:
        image = cv2.resize(image, (256, 256))
        image = preprocess_input(image.astype('float32'))
        return np.expand_dims(image, axis=0)
    else:
        raise ValueError("Invalid image format")

def main():
    st.title("🩸 Blood Group Detection from Fingerprint")
    
    # Load model and metrics
    model = load_model()
    metrics = load_model_metrics()
    
    if model is None:
        return
    
    # Display model metrics in a nice format
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Accuracy", f"{metrics['accuracy']:.2f}%" if metrics else "N/A")
    with col2:
        st.metric("Validation Accuracy", f"{metrics['val_accuracy']:.2f}%" if metrics else "N/A")
    with col3:
        st.metric("Training Loss", f"{metrics['loss']:.4f}" if metrics else "N/A")
    
    # Display performance graph
    if os.path.exists('static/model_performance.png'):
        st.subheader("📈 Model Performance")
        st.image('static/model_performance.png')
    
    # File upload section
    st.subheader("🔍 Upload Fingerprint Image")
    uploaded_file = st.file_uploader("Choose a fingerprint image...", type=["bmp", "jpg", "png"])
    
    if uploaded_file is not None:
        try:
            # Create two columns for image and prediction
            col1, col2 = st.columns(2)
            
            # Display uploaded image with reduced size
            image = Image.open(uploaded_file)
            with col1:
                # Add a container with custom width
                st.markdown("""
                    <style>
                        .image-container {
                            max-width: 400px;
                            margin: auto;
                        }
                    </style>
                """, unsafe_allow_html=True)
                
                with st.container():
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(image, caption="Uploaded Fingerprint", width=400)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Make prediction
            with st.spinner('Analyzing fingerprint...'):
                processed_image = preprocess_image(image)
                prediction = model.predict(processed_image)
                
                blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
                result = blood_groups[np.argmax(prediction)]
                confidence = float(np.max(prediction) * 100)
                
                # Display prediction results
                with col2:
                    st.markdown(f"""
                    <div class="success">
                        <h2 style="color: #00FF00;">Prediction Results</h2>
                        <h3>Blood Group: {result}</h3>
                        <p>Confidence: {confidence:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Progress bar for confidence
                    st.progress(confidence/100.0)
                    
                    # Display probability distribution
                    st.subheader("Probability Distribution")
                    probs = prediction[0].astype(float)
                    for bg, prob in zip(blood_groups, probs):
                        st.progress(prob)
                        st.write(f"{bg}: {prob*100:.2f}%")
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.info("Please ensure you upload a valid fingerprint image in BMP, JPG, or PNG format.")

if __name__ == "__main__":
    main()