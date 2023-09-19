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
warnings.filterwarnings("ignore", category=FutureWarning)



iris_testing_df = pd.read_excel('testing_dataset.xlsx')
X = iris_testing_df.iloc[:,:-1]
y=iris_testing_df.iloc[:,-1]
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


def feature_testing1():
    path = 'updated_iris_model.joblib'
    mock_model = joblib.load(path)
    correct = 0
    for i in range(len(X)):
        f1 = X.iloc[i,0]
        f2 = X.iloc[i,1]
        f3 = X.iloc[i,2]
        f4 = X.iloc[i,3]

        prediction = iris_model_prediction(mock_model, f1,f2,f3,f4)
        print("\n\n")
        
        
        if prediction==y_encoded[0]:
            correct+=1
    print(f"Correct = {correct}\nTotal = {len(X)}")

feature_testing1()