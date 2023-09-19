import unittest
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from helper import Model, load_iris_classes, preprocess_input_image, iris_model_prediction, flower_model_prediction,load_iris_model

class TestIrisAndFlowerClassification(unittest.TestCase):
    def test_load_iris_classes(self):
        iris_classes = load_iris_classes()
        self.assertEqual(len(iris_classes), 3)  # Check if there are 3 classes
        
    def test_preprocess_input_image(self):
        image = np.random.random((256, 256, 3))  # Creating a random image
        preprocessed_image = preprocess_input_image(image)
        self.assertEqual(preprocessed_image.shape, (128, 128, 3))  # Check if image is resized correctly
        self.assertTrue(np.all(preprocessed_image >= 0) and np.all(preprocessed_image <= 1))  # Check pixel values range
        
    def test_iris_model_prediction(self):
        path = 'iris_model.h5'
        mock_model = load_iris_model(path)
        
        sepal_length = 5.1
        sepal_width = 3.5
        petal_length = 1.4
        petal_width = 0.2
        
        prediction = iris_model_prediction(mock_model, sepal_length, sepal_width, petal_length, petal_width)
        self.assertIn(prediction, [0, 1, 2])  # Check if prediction is one of the valid classes
        
    def test_flower_model_prediction(self):
        mock_model = tf.keras.models.load_model('flower_model.h5')  # Replace with your flower model filename
        
        image = np.random.random((256, 256, 3))  # Creating a random image
        prediction = flower_model_prediction(mock_model, image)
        self.assertIn(prediction, [0, 1, 2])  # Check if prediction is one of the valid classes

if __name__ == '__main__':
    unittest.main()
