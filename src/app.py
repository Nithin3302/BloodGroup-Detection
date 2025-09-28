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

# Apply custom CSS for red theme
st.markdown("""
    <style>
        .stApp {
            background-color: #000000;  /* Changed to pure black */
            color: #FAFAFA;
        }
        .stButton>button {
            background-color: #FF0000;
            color: white;
            padding: 0.5rem !important;
        }
        .stProgress .st-bo {
            background-color: #FF0000;
        }
        .success {
            padding: 0.5rem;
            border-radius: 0.25rem;
            background-color: rgba(255, 0, 0, 0.1);
            border: 1px solid #FF0000;
            margin: 0.5rem 0;
        }
        h1 {
            color: #FF0000 !important;
            font-size: 2rem !important;
            margin: 0.5rem 0 !important;
            text-align: center !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        h2 {
            color: #FF0000 !important;
            font-size: 1.2rem !important;
            margin: 0.4rem 0 !important;
        }
        h3 {
            color: #FF0000 !important;
            font-size: 1rem !important;
            margin: 0.3rem 0 !important;
        }
        .metric-container {
            background: rgba(255, 0, 0, 0.1);
            padding: 0.5rem;
            border-radius: 0.5rem;
            border: 1px solid rgba(255, 0, 0, 0.2);
            text-align: center;
            margin: 0.2rem;
        }
        .metric-value {
            font-size: 1.2rem;
            color: #FF0000;
            font-weight: bold;
        }
        .metric-label {
            font-size: 0.7rem;
            color: #FFB6B6;
            text-transform: uppercase;
        }
        .stProgress {
            height: 0.3rem !important;
        }
        
        /* Remove any default padding/margins around file uploader */
        .stFileUploader {
            padding: 0 !important;
            margin: 0 !important;
        }
        
        /* Hide the default file uploader border */
        .uploadedFile {
            border: none !important;
            background: none !important;
        }
        
        /* Custom styling for file uploader */
        .css-1kovmik {
            background-color: transparent !important;
        }
        
        .css-1kmuo5j {
            background-color: rgba(255, 0, 0, 0.1) !important;
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
    st.title("🩸 Blood Group Detection")
    
    # Load model and metrics
    model = load_model()
    metrics = load_model_metrics()
    
    if model is None:
        return
    
    # Redesigned metrics display
    st.markdown("""
        <div style='display: flex; justify-content: center; gap: 1rem; margin-bottom: 1rem;'>
            <div class='metric-container'>
                <div class='metric-value'>{:.1f}%</div>
                <div class='metric-label'>Accuracy</div>
            </div>
            <div class='metric-container'>
                <div class='metric-value'>{:.1f}%</div>
                <div class='metric-label'>Validation</div>
            </div>
            <div class='metric-container'>
                <div class='metric-value'>{:.2f}</div>
                <div class='metric-label'>Loss</div>
            </div>
        </div>
    """.format(
        metrics['accuracy'] if metrics else 0,
        metrics['val_accuracy'] if metrics else 0,
        metrics['loss'] if metrics else 0
    ), unsafe_allow_html=True)
    
    # Smaller performance graph
    if os.path.exists('static/model_performance.png'):
        with st.expander("📈 Model Performance", expanded=False):
            st.image('static/model_performance.png')
    
    # File upload with minimal styling
    uploaded_file = st.file_uploader("Upload Fingerprint", type=["bmp", "jpg", "png"], 
                                   label_visibility="collapsed")
    
    if uploaded_file is not None:
        try:
            col1, col2 = st.columns(2)
            
            # Display image without extra border div
            image = Image.open(uploaded_file)
            with col1:
                st.image(image, caption="Fingerprint", width=200)
            
            # Make prediction
            with st.spinner('Analyzing...'):
                processed_image = preprocess_image(image)
                prediction = model.predict(processed_image)
                blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
                result = blood_groups[np.argmax(prediction)]
                confidence = float(np.max(prediction) * 100)
                
                # Enhanced prediction display
                with col2:
                    st.markdown(f"""
                    <div class="success" style='text-align: center;'>
                        <h2 style='color: #FF0000; margin: 0;'>Blood Group: {result}</h2>
                        <div style='font-size: 0.9rem; color: #FFB6B6;'>Confidence: {confidence:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.progress(confidence/100.0)
                    
                    # Compact probability distribution
                    with st.expander("Detailed Analysis"):
                        probs = prediction[0].astype(float)
                        for bg, prob in zip(blood_groups, probs):
                            col1, col2 = st.columns([1, 4])
                            with col1:
                                st.markdown(f"<div style='color: #FFB6B6;'>{bg}:</div>", unsafe_allow_html=True)
                            with col2:
                                st.progress(prob)
                                st.markdown(f"<div style='font-size: 0.8rem; color: #FFB6B6; text-align: right;'>{prob*100:.1f}%</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()