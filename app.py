import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np

st.header('Image Classification Using CNNs.')
model = load_model('C:\\Image Classification Using CNNs\\image_classify.keras')
data_cat = ['apple',
            'banana',
            'beetroot',
            'bell pepper',
            'cabbage',
            'capsicum',
            'carrot',
            'cauliflower',
            'chilli pepper',
            'corn',
            'cucumber',
            'eggplant',
            'garlic',
            'ginger',
            'grapes',
            'jalepeno',
            'kiwi',
            'lemon',
            'lettuce',
            'mango',
            'onion',
            'orange',
            'paprika',
            'pear',
            'peas',
            'pineapple',
            'pomegranate',
            'potato',
            'raddish',
            'soy beans',
            'spinach',
            'sweetcorn',
            'sweetpotato',
            'tomato',
            'turnip',
            'watermelon']
img_height = 180
img_width = 180
image = st.text_input('Enter Image Name:', 'Apple.jpg')

image_load = tf.keras.utils.load_img(image, target_size=(img_height, img_width))
img_arr = tf.keras.utils.array_to_img(image_load)
img_bat = tf.expand_dims(img_arr, 0)

predict = model.predict(img_bat)

score = tf.nn.softmax(predict)
st.image(image, width=200)
st.write('Veg/Fruit in Image: The object in the image is identified as ' + data_cat[np.argmax(score)] + ".")
st.write('Accuracy: The accuracy of this classification is ' + str(np.max(score) * 100) + ".")
st.header("**The image classification model successfully identified the object with an accuracy.**")
