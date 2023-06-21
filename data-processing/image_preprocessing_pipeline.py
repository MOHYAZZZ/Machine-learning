"""
This script defines a TensorFlow Sequential model for image preprocessing. 
The model is designed to be used as a preprocessing pipeline for image data, applying various transformations and augmentations to enhance the training dataset. 
The pipeline includes resizing, rescaling, random flips, rotations, contrast adjustments, cropping, zooming, translation, and brightness modifications. 
The `get_image_preprocessing_pipeline` function returns the built model based on the specified input size and resize dimensions.
"""


import tensorflow as tf
from tensorflow.keras import layers

def get_image_preprocessing_pipeline(input_size, resize):
    model = tf.keras.Sequential([
        layers.Resizing(height=resize, width=resize),
        layers.Rescaling(1./255),
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.1),
        layers.RandomContrast(0.1),
        layers.RandomCrop(int(resize*.9), int(resize*.9), 3),
        layers.RandomZoom(height_factor=0.1, width_factor=0.1),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomBrightness(factor=0.1),
    ])
    model.build(input_shape=(input_size, input_size, 3))
    return model
