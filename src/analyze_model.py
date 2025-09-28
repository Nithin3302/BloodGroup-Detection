import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

def analyze_model_performance(model_path='models/blood_group_model.h5'):
    """Analyze and display model performance metrics"""
    model = tf.keras.models.load_model(model_path)
    
    # Get training history if available
    history = model.history.history if hasattr(model, 'history') else None
    
    if history:
        # Plot accuracy and loss curves
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('static/model_performance.png')
        plt.close()
        
        # Get final metrics
        final_acc = history['val_accuracy'][-1]
        return {
            'accuracy': final_acc,
            'history': history
        }
    return None

if __name__ == "__main__":
    results = analyze_model_performance()
    if results:
        print(f"Model Accuracy: {results['accuracy']*100:.2f}%")