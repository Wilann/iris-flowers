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
import joblib

from helper import load_iris_model,load_iris_classes,preprocess_input_image,iris_model_prediction,flower_model_prediction


# Starting PopUp Window
show_info = st.button("Click Here to get information related to the project", key='a1')

if show_info:
    def header(url):
     st.markdown(f'<p style="background-color:#0066cc;color:#33ff33;font-size:24px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)

    header('This is an iris and flower classification application. The application includes two main features.\nOne is iris classification using numerical feature values and the other is flower classification using images.\nFrom the left side of the window, you can select from the two features of this application (iris and flower classification)\nEnter the feature values in the boxes provided and click the Predict Button')
    
    # Add a "Close" button to hide the information
    if st.button("Close"):
        show_info = False

# Iris model
path = 'iris_model.h5' # For deep learning model
path2 = 'updated_iris_model.joblib' # For KNN model
iris_model2 = joblib.load(path2)
info_about_flower = [
    'Bellis perennis, the daisy, is a European species of the family Asteraceae, often considered the archetypal species of the name daisy. To distinguish this species from other plants known as daisies, it is sometimes qualified as common daisy, lawn daisy or English daisy.  \nScientific name: Bellis perennis  \nFamily: Asteraceae  \nKingdom: Plantae  \nOrder: Asterales\nMore Information: https://kellogggarden.com/blog/gardening/how-to-plant-grow-and-care-for-daisy-flowers/#:~:text=Watering%20and%20Feeding%20Daisies,sun%20damage%20to%20tender%20petals.' ,
    'Taraxacum is a large genus of flowering plants in the family Asteraceae, which consists of species commonly known as dandelions. The scientific and hobby study of the genus is known as taraxacology.  \nScientific name: Taraxacum  \nFamily: Asteraceae  \nKingdom: Plantae  \nOrder: Asterales  \nSubfamily: Cichorioideae\nSubtribe: Crepidinae\nMore Information: https://www.masterclass.com/articles/growing-dandelions-explained',
    'A rose is either a woody perennial flowering plant of the genus Rosa, in the family Rosaceae, or the flower it bears. There are over three hundred species and tens of thousands of cultivars  \nScientific name: Rosa  \nHigher classification: Rosoideae\nMore Information: https://gardens.si.edu/learn/blog/tips-for-growing-healthy-roses/#:~:text=Give%20them%20what%20they%20need,not%20to%20the%20leaf%20surface.',
    'Helianthus is a genus comprising about 70 species of annual and perennial flowering plants in the daisy family Asteraceae commonly known as sunflowers. Except for three South American species, the species of Helianthus are native to North America and Central America. The best-known species is the common sunflower  \nScientific name: Helianthus  \nFamily: Asteraceae  \nFamily: Asteraceae  \nKingdom: Plantae  \nOrder: Asterales  \nTribe: Heliantheae\nMore Information: https://www.southernliving.com/garden/flowers/how-to-grow-sunflowers#:~:text=North%20America-,Sunflower%20Care,enough%20drainage%20and%20loose%20soil.',
    'Tulips are a genus of spring-blooming perennial herbaceous bulbiferous geophytes. The flowers are usually large, showy and brightly coloured, generally red, pink, yellow, or white. They often have a different coloured blotch at the base of the tepals, internally.  \nScientific name: Tulipa  \nFamily: Liliaceae  \nKingdom: Plantae  \nOrder: Liliales  \nTribe: Lilieae\nMore Information: https://www.thespruce.com/tulips-planting-and-growing-tulips-1402137#:~:text=Tulips%20require%20full%20sun%2C%20neutral,in%20a%20cold%2Dweather%20zone.'
]

info_about_iris = [
    'Iris setosa, the bristle-pointed iris, is a species of flowering plant in the genus Iris of the family Iridaceae, it belongs the subgenus Limniris and the series Tripetalae  \nIris setosa, the bristle-pointed iris, is a species of flowering plant in the genus Iris of the family Iridaceae, it belongs the subgenus Limniris and the series Tripetalae  \nIris setosa, the bristle-pointed iris, is a species of flowering plant in the genus Iris of the family Iridaceae, it belongs the subgenus Limniris and the series Tripetalae  \nRank: Species  \nFamily: Iridaceae  \nKingdom: Plantae\nMore Information: https://mygardenlife.com/plant-library/dwarf-arctic-iris-iris-setosa-var-arctica#:~:text=Incorporate%20fertilizer%20into%20the%20soil,easily%20damaged%20by%20early%20frosts.',
    'Iris versicolor is also commonly known as the blue flag, harlequin blueflag, larger blue flag, northern blue flag, and poison flag, plus other variations of these names, and in Britain and Ireland as purple iris. It is a species of Iris native to North America, in the Eastern United States and Eastern Canada.  \nScientific name: Iris versicolor  \nRank: Species  \nFamily: Iridaceae\nMore Information: https://www.thespruce.com/northern-blue-flag-native-iris-4125732',
    'Iris virginica, with the common name Virginia blueflag, Virginia iris, great blue flag, or southern blue flag, is a perennial species of flowering plant in the Iridaceae family, native to central and eastern North America  \nScientific name: Iris virginica  \nHigher classification: Irises  \nFamily: Iridaceae  \nRank: Species\nMore Information: https://www.allaboutgardening.com/southern-blue-flag/'
]



# Create an instance of your model
iris_model = load_iris_model(path)


# Flower model
flower_model = tf.keras.models.load_model('flower_model.h5')



# Load Iris dataset
iris_classes = load_iris_classes()

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
        prediction = iris_model_prediction(iris_model2,sepal_length, sepal_width, petal_length, petal_width)
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

        predictions = flower_model_prediction(flower_model,image)
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
        elif predicted_class == class_names[4]:
            flower_desc = info_about_flower[4]
                

        st.write("Prediction: ",predicted_class)
        st.write("Description: ",flower_desc)
        
       

