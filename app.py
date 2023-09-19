import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

model = tf.keras.models.load_model('digit_recog_mod.h5')

def predict_number(image):
    processed_image = preprocess_image(image)
    st.image(processed_image, caption='Processed Image.', use_column_width=True)  # Add this line to display the processed image
    prediction = model.predict(np.array([processed_image]))
    st.write('Raw Predictions:', prediction)  # Add this line to display the raw predictions
    predicted_number = np.argmax(prediction)
    st.write('Predicted Number:', predicted_number)
    return predicted_number

def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((28, 28))
    img = img.convert('L')
    processed_image = np.array(img) / 255.0
    return processed_image

st.title('Handwritten Digit Recognition')

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    predicted_number = predict_number(uploaded_file)
