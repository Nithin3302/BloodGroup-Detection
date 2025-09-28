from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def create_model():
    """Create enhanced model architecture"""
    pretrained_model = ResNet50(
        input_shape=(256,256,3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    
    # Fine-tune last few layers
    for layer in pretrained_model.layers[:-10]:
        layer.trainable = False
    
    inputs = pretrained_model.input
    x = pretrained_model.output
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(8, activation='softmax')(x)
    
    return Model(inputs=inputs, outputs=outputs)