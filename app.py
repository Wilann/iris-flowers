import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from tensorflow import keras
import tensorflow as tf
import keras.models
from sklearn.datasets import load_iris
import cv2
from streamlit_modal import Modal


# Starting PopUp Window
show_info = st.button("Click Here to get information related to the project", key='a1')

if show_info:
    st.write('This is an iris and flower classification application. The application includes two main features.')
    st.write('One is iris classification using numerical feature values and the other is flower classification using images.')
    st.write('From the left side of the window, you can select from the two features of this application (iris and flower classification)')
    st.write('Enter the feature values in the boxes provided and click the Predict Button')
    
    # Add a "Close" button to hide the information
    if st.button("Close"):
        show_info = False

# Iris model
path = 'iris_model.h5'
info_about_flower = [
    'Bellis perennis, the daisy, is a European species of the family Asteraceae, often considered the archetypal species of the name daisy. To distinguish this species from other plants known as daisies, it is sometimes qualified as common daisy, lawn daisy or English daisy.  \nScientific name: Bellis perennis  \nFamily: Asteraceae  \nKingdom: Plantae  \nOrder: Asterales' ,
    'Taraxacum is a large genus of flowering plants in the family Asteraceae, which consists of species commonly known as dandelions. The scientific and hobby study of the genus is known as taraxacology.  \nScientific name: Taraxacum  \nFamily: Asteraceae  \nKingdom: Plantae  \nOrder: Asterales  \nSubfamily: Cichorioideae\nSubtribe: Crepidinae',
    'A rose is either a woody perennial flowering plant of the genus Rosa, in the family Rosaceae, or the flower it bears. There are over three hundred species and tens of thousands of cultivars  \nScientific name: Rosa  \nHigher classification: Rosoideae',
    'Helianthus is a genus comprising about 70 species of annual and perennial flowering plants in the daisy family Asteraceae commonly known as sunflowers. Except for three South American species, the species of Helianthus are native to North America and Central America. The best-known species is the common sunflower  \nScientific name: Helianthus  \nFamily: Asteraceae  \nFamily: Asteraceae  \nKingdom: Plantae  \nOrder: Asterales  \nTribe: Heliantheae',
    'Tulips are a genus of spring-blooming perennial herbaceous bulbiferous geophytes. The flowers are usually large, showy and brightly coloured, generally red, pink, yellow, or white. They often have a different coloured blotch at the base of the tepals, internally.  \nScientific name: Tulipa  \nFamily: Liliaceae  \nKingdom: Plantae  \nOrder: Liliales  \nTribe: Lilieae'
]

info_about_iris = [
    'Iris setosa, the bristle-pointed iris, is a species of flowering plant in the genus Iris of the family Iridaceae, it belongs the subgenus Limniris and the series Tripetalae  \nIris setosa, the bristle-pointed iris, is a species of flowering plant in the genus Iris of the family Iridaceae, it belongs the subgenus Limniris and the series Tripetalae  \nIris setosa, the bristle-pointed iris, is a species of flowering plant in the genus Iris of the family Iridaceae, it belongs the subgenus Limniris and the series Tripetalae  \nRank: Species  \nFamily: Iridaceae  \nKingdom: Plantae',
    'Iris versicolor is also commonly known as the blue flag, harlequin blueflag, larger blue flag, northern blue flag, and poison flag, plus other variations of these names, and in Britain and Ireland as purple iris. It is a species of Iris native to North America, in the Eastern United States and Eastern Canada.  \nScientific name: Iris versicolor  \nRank: Species  \nFamily: Iridaceae',
    'Iris virginica, with the common name Virginia blueflag, Virginia iris, great blue flag, or southern blue flag, is a perennial species of flowering plant in the Iridaceae family, native to central and eastern North America  \nScientific name: Iris virginica  \nHigher classification: Irises  \nFamily: Iridaceae  \nRank: Species'
]

class ANNModel(keras.Model):
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

# Create an instance of your model
iris_model = ANNModel()

_ = iris_model(tf.keras.Input(shape=(4,)))

# Load the saved model weights
iris_model.load_weights(path)

# Flower model
flower_model = tf.keras.models.load_model('flower_model.h5')



# Load Iris dataset
iris = load_iris()
iris_classes = iris.target_names


def preprocess_input(image):
    image_size = (128, 128)
    image = cv2.resize(image, image_size)
    image = image / 255.0  
    return image





# Streamlit app
st.title("Iris and Flower Classification")

# Classification options
classification_option = st.sidebar.selectbox("Choose a classification option:",
                                             ("Iris Classification", "Flower Classification"))

if classification_option == "Iris Classification":
    st.header("Iris Classification")
    st.write("Please enter the sepal and petal measurements:")

    sepal_length = st.number_input("Sepal Length")
    sepal_width = st.number_input("Sepal Width")
    petal_length = st.number_input("Petal Length")
    petal_width = st.number_input("Petal Width")
    if st.button('Predict'):

        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = np.argmax(iris_model.predict(features))
        iris_desc = info_about_iris[prediction]

        st.write("Prediction: ", iris_classes[prediction])
        st.write("Description: ", iris_desc)

elif classification_option == "Flower Classification":
    st.header("Flower Classification")
    st.write("Please upload an image for classification:")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        # Flower classification prediction
        image = np.array(image)
        image = preprocess_input(image)
        image = np.expand_dims(image, axis=0)

        preds = flower_model.predict(image)
        predictions = np.argmax(preds[0])
        class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
        predicted_class = class_names[predictions]
        flower_desc = ''
        if predicted_class == class_names[0]:
            flower_desc = info_about_flower[0]
        elif predicted_class == class_names[1]:
            flower_desc = info_about_flower[1]
        elif predicted_class == class_names[2]:
            flower_desc = info_about_flower[2]
        elif predicted_class == class_names[3]:
            flower_desc = info_about_flower[3]
                

        st.write("Prediction: ",predicted_class)
        st.write("Description: ",flower_desc)
        
       

