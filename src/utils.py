from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import tensorflow as tf

def setup_callbacks(model_path):
    """Setup training callbacks"""
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        min_delta=0,
        patience=2,
        mode='auto'
    )
    
    return [checkpoint, early_stopping]

def load_trained_model(model_path):
    """Load trained model if it exists"""
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        return tf.keras.models.load_model(model_path)
    return None