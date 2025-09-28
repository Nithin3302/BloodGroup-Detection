import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

def prepare_dataset(data_path):
    """Prepare dataset for training"""
    # Get file paths and labels
    data_path = os.path.join(data_path, 'dataset_blood_group')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please place the dataset in the data folder.")
    
    print("Loading dataset from:", data_path)
    filepaths = list(glob.glob(os.path.join(data_path, '**/*.*')))
    labels = [os.path.split(os.path.split(x)[0])[1] for x in filepaths]
    
    print(f"Found {len(filepaths)} images in {len(set(labels))} classes.")
    
    # Create dataframe
    data = pd.DataFrame({
        'Filepath': filepaths,
        'Label': labels
    })
    
    # Split data
    train, valid = train_test_split(data, test_size=0.20, random_state=42)
    
    # Create generators
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        shear_range=0.2,
        zoom_range=0.2,
        brightness_range=[0.8,1.2],
        fill_mode='nearest'
    )
    
    valid_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )
    
    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train,
        x_col='Filepath',
        y_col='Label',
        target_size=(256,256),
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=42
    )
    
    valid_gen = valid_datagen.flow_from_dataframe(
        dataframe=valid,
        x_col='Filepath',
        y_col='Label',
        target_size=(256,256),
        class_mode='categorical',
        batch_size=32,
        shuffle=False,
        seed=42
    )
    
    return train_gen, valid_gen