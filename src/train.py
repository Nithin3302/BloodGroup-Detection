import argparse
import os
import sys
from model import create_model
from data_processing import prepare_dataset
from utils import setup_callbacks

def train_model(args):
    """Train the blood group detection model"""
    # Check if model exists
    if not args.retrain and os.path.exists(args.model_path):
        print(f"Model already exists at {args.model_path}")
        return
    
    # Prepare dataset
    train_gen, valid_gen = prepare_dataset(args.data_path)
    
    # Create and compile model
    model = create_model()
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Setup callbacks
    callbacks = setup_callbacks(args.model_path)
    
    # Train model
    history = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=args.epochs,
        callbacks=callbacks
    )
    
    # Save final model
    model.save(args.model_path)
    print(f"Model saved to {args.model_path}")

def main():
    parser = argparse.ArgumentParser(description='Train Blood Group Detection Model')
    parser.add_argument('--data_path', type=str, default='data',
                       help='Path to dataset directory')
    parser.add_argument('--model_path', type=str, 
                       default='models/blood_group_model.h5',
                       help='Path to save/load model')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs to train')
    parser.add_argument('--retrain', action='store_true',
                       help='Force retraining even if model exists')
    
    args = parser.parse_args()
    train_model(args)

if __name__ == "__main__":
    main()