import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(page_title="Diabetic Retinopathy Detection", page_icon="👁️", layout="centered")

st.title("Diabetic Retinopathy Detection System")
st.write("Upload a retinal fundus photograph to analyze for signs of Diabetic Retinopathy.")

IMG_SIZE = (224, 224) 

@st.cache_resource
def load_keras_model():
    try:
        model = tf.keras.models.load_model('model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model file: {e}")
        return None

model = load_keras_model()

uploaded_file = st.file_uploader("Choose a retinal image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Retinal Scan', use_container_width=True)
    
    st.write("")
    if st.button('Analyze Image'):
        if model is None:
            st.error("Model is not loaded properly. Check your file name.")
        else:
            with st.spinner('Running deep learning diagnosis pipeline...'):

                img = image.convert('RGB')
                img = img.resize(IMG_SIZE)
                img_array = np.array(img)
                
                img_array = img_array / 255.0
                
                img_array = np.expand_dims(img_array, axis=0)
                
                predictions = model.predict(img_array)
                
                probability = predictions[0][0]
                
                st.markdown("---")
                st.subheader("Results:")
                
                if probability >= 0.5:
                    st.error(f"**Diabetic Retinopathy Detected**")
                    st.write(f"Confidence score: {probability * 100:.2f}%")
                else:
                    st.success(f"**Normal Retina (No DR Detected)**")
                    st.write(f"Confidence score: {(1 - probability) * 100:.2f}%")
                    
st.markdown("---")
st.caption("Disclaimer: This tool is built for educational/project demonstration use and does not constitute official clinical medical advice.")
