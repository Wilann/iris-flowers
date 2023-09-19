import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from tensorflow import keras
import tensorflow as tf
import keras.models
from sklearn.datasets import load_iris
import cv2

class Model(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_layer = keras.layers.Flatten(input_shape=(4,))
        self.hidden1 = keras.layers.Dense(128, activation='relu')
        self.hidden2 = keras.layers.Dense(64, activation='relu')
        self.hidden3 = keras.layers.Dense(32, activation='relu')
        self.output_layer = keras.layers.Dense(3, activation='softmax')
        self.dropout_layer = keras.layers.Dropout(rate=0.2)
    
    def call(self, input, training=None):
        input_layer = self.input_layer(input)
        input_layer = self.dropout_layer(input_layer)
        hidden1 = self.hidden1(input_layer)
        hidden1 = self.dropout_layer(hidden1, training=training)
        hidden2 = self.hidden2(hidden1)
        hidden2 = self.dropout_layer(hidden2, training=training)
        hidden3 = self.hidden3(hidden2)
        hidden3 = self.dropout_layer(hidden3, training=training)
        output_layer = self.output_layer(hidden3)
        return output_layer

def load_iris_model(path):
    iris_model = Model()

    _ = iris_model(tf.keras.Input(shape=(4,)))

    # Load the saved model weights
    iris_model.load_weights(path)
    return iris_model

def load_iris_classes():
    try:
        iris = load_iris() # Load Iris dataset
        iris_classes = iris.target_names
        return iris_classes
    except Exception as e:
        print(f'Error occured in "load_iris_classes" function.\n Error = {e}')
    

def preprocess_input_image(image):
    try:
        image_size = (128, 128)
        image = cv2.resize(image, image_size)
        image = image / 255.0  
        return image
    except Exception as e:
        print(f'Error occured in "preprocess_input_image" function.\n Error = {e}')



def iris_model_prediction(iris_model,sepal_length, sepal_width, petal_length, petal_width):
    try:
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = np.argmax(iris_model.predict(features))
        return prediction
    except Exception as e:
        print(f'Error occured in "iris_model_prediction" function.\n Error = {e}')

        

def flower_model_prediction(flower_model,image):
    try:
        image = preprocess_input_image(image)
        image = np.expand_dims(image, axis=0)
        preds = flower_model.predict(image)
        predictions = np.argmax(preds[0])
        return predictions
    except Exception as e:
        print(f'Error occured in "flower_model_prediction" function.\n Error = {e}')
    

