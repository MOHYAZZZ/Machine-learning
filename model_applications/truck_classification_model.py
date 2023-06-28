"""
This script defines a function to create a simple Convolutional Neural Network (CNN) model
for classifying images as either trucks or not trucks (binary classification).

The model architecture consists of three sets of Conv2D and MaxPooling2D layers for feature 
extraction from images, followed by a Flatten layer and two Dense layers for classification.
The model uses 'relu' activation function in Conv2D and first Dense layer, while 'sigmoid'
activation function in the final Dense layer for binary classification. 

The model is compiled with Adam optimizer and binary cross entropy loss, which is appropriate 
for binary classification tasks. The learning rate for the optimizer is set to 0.01. The model's 
performance is evaluated based on binary accuracy.

"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses

def classify_trucks():
    model = models.Sequential()
    model.add(layers.Conv2D(16, 3, activation='relu', input_shape=(224, 224, 3), kernel_initializer='he_normal'))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Conv2D(32, 3, activation='relu', kernel_initializer='he_normal'))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Conv2D(64, 3, activation='relu', kernel_initializer='he_normal'))
    model.add(layers.Flatten())
    model.add(layers.Dense(20, activation='relu', kernel_initializer='he_normal'))
    model.add(layers.Dense(1, activation='sigmoid', kernel_initializer='glorot_normal'))

    model.compile(optimizer=optimizers.Adam(learning_rate=0.01), 
                  loss=losses.BinaryCrossentropy(), 
                  metrics=['binary_accuracy'])

    return model
