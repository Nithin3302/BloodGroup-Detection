import tensorflow as tf
import cv2
import numpy as np
import os
from tensorflow.keras.applications.resnet50 import preprocess_input

def predict_blood_group(image_path, model_path='models/blood_group_model.h5'):
    """Predict blood group from fingerprint image"""
    # Convert to absolute path and normalize
    image_path = os.path.abspath(image_path)
    print(f"Looking for image at: {image_path}")
    
    # Debug info
    if os.path.exists(os.path.dirname(image_path)):
        print(f"Directory exists: {os.path.dirname(image_path)}")
        print("Files in directory:")
        for f in os.listdir(os.path.dirname(image_path)):
            print(f"  - {f}")
    else:
        print(f"Directory does not exist: {os.path.dirname(image_path)}")
    
    # Check if image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading image...")
    model = tf.keras.models.load_model(model_path)
    
    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    print(f"Image shape: {img.shape}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    
    # Predict
    prediction = model.predict(img)
    blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
    result = blood_groups[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    print(f"\nPredicted Blood Group: {result}")
    print(f"Confidence: {confidence:.2f}%")
    
    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help='Path to fingerprint image')
    parser.add_argument('--model_path', default='models/blood_group_model.h5',
                       help='Path to trained model')
    
    args = parser.parse_args()
    try:
        result = predict_blood_group(args.image_path, args.model_path)
    except Exception as e:
        print(f"Error: {str(e)}")