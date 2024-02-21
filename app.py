import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
  model = tf.keras.models.load_model('C:\\Users\\LENOVO\\projectinno\\potato-disease-classification\\potatoes.h5')
  return model

with st.spinner('Model is being loaded..'):
  model = load_model()

st.write("# **Potato Disease Classification**")
         

file = st.file_uploader("Upload the image to be classified U0001F447", type=["jpg", "png"])
st.set_option('deprecation.showfileUploaderEncoding', False)

def upload_predict(upload_image, model):
    size = (180, 180)
    image = ImageOps.fit(upload_image, size, Image.LANCZOS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resize = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    img_reshape = img_resize[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    
    # Get the class with the highest probability
    predicted_class = np.argmax(prediction)
    
    # Define your class names based on your model
    class_names =['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
    
    predicted_class_name = class_names[predicted_class]
    confidence_percentage = np.max(prediction) * 100
    
    return predicted_class_name, confidence_percentage

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predicted_class, confidence = upload_predict(image, model)
    st.write(f"The image is classified as {predicted_class} with a confidence of {confidence:.2f}%")


 