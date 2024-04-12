# Load necessary libraries
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# Load the saved model
model = load_model('image_classify.keras')

# Define the categories
data_cat = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']

# Define the image path
image_path = 'corn.jpg'

# Load and preprocess the image
image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
image_array = tf.keras.utils.img_to_array(image)
image_batch = np.expand_dims(image_array, axis=0)

# Make predictions
predictions = model.predict(image_batch)

# Get the predicted category
predicted_category = data_cat[np.argmax(predictions)]

# Get the probability score
probability_score = np.max(predictions) * 100

# Print the results
print(f"The predicted category for the image '{image_path}' is '{predicted_category}' with an accuracy of {probability_score:.2f}%.")

# Plot the prediction result
plt.figure(figsize=(6, 4))
plt.bar(predicted_category, probability_score, color='blue')
plt.xlabel('Category')
plt.ylabel('Probability (%)')
plt.title('Image Classification Result')
plt.ylim(0, 100)
plt.grid(axis='y')
plt.show()