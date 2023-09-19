import unittest
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from helper import Model, load_iris_classes, preprocess_input_image, iris_model_prediction, flower_model_prediction,load_iris_model
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
import os
warnings.filterwarnings("ignore", category=FutureWarning)

main_folder = r'flowers\testing_flower'

def feature_testing2():

    # Get the list of subfolders (categories) in the main folder
    categories = os.listdir(main_folder)

    # Flower model
    flower_model = tf.keras.models.load_model('flower_model.h5')

    # Get the list of subfolders (categories) in the main folder
    categories = os.listdir(main_folder)
    print(f'categ = {categories}')

    for category in categories:
        category_folder = os.path.join(main_folder, category)
        
        if os.path.isdir(category_folder):  # Check if the path is a directory
            # List all image files in the subfolder
            image_files = [f for f in os.listdir(category_folder) if f.endswith('.jpg') or f.endswith('.png')]

            for image_file in image_files:
                image_path = os.path.join(category_folder, image_file)
                image = cv2.imread(image_path)

                prediction = flower_model_prediction(flower_model,image)
                

                print('ok')
                print(f"Image: {image_file} | Predicted Class: {prediction} | Category: {category}")

feature_testing2()