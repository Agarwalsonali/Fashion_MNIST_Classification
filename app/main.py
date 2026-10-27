import json
import streamlit as st
import tensorflow as tf
import h5py
from PIL import Image
import numpy as np
import os

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(
    working_dir, "trained_model", "trained_fashion_mnist_model.h5"
)


def _convert_keras_3_config(value):
    if isinstance(value, dict):
        if value.get("class_name") == "DTypePolicy":
            return value["config"]["name"]
        return {
            key: _convert_keras_3_config(item)
            for key, item in value.items()
            if key != "quantization_config"
        }
    if isinstance(value, list):
        return [_convert_keras_3_config(item) for item in value]
    return value


def load_model_compatibly(path):
    with h5py.File(path, "r") as model_file:
        keras_version = model_file.attrs.get("keras_version", "")

    if isinstance(keras_version, bytes):
        keras_version = keras_version.decode("utf-8")

    if not str(keras_version).startswith("3."):
        return tf.keras.models.load_model(path, compile=False)

    with h5py.File(path, "r") as model_file:
        model_config = model_file.attrs["model_config"]
        if isinstance(model_config, bytes):
            model_config = model_config.decode("utf-8")

    config = _convert_keras_3_config(json.loads(model_config))
    for layer in config["config"]["layers"]:
        if layer.get("class_name") == "InputLayer":
            layer_config = layer["config"]
            if "batch_shape" in layer_config:
                layer_config["batch_input_shape"] = layer_config.pop("batch_shape")
            layer_config.pop("optional", None)

    model = tf.keras.models.model_from_json(json.dumps(config))
    model.load_weights(path)
    return model


model = load_model_compatibly(model_path)

# Define class labels for Fashion MNIST dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Function to preprocess the uploaded image
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((28, 28))
    img = img.convert('L')  # Convert to grayscale
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1, 28, 28, 1))
    return img_array

# Streamlit App
st.title('Fashion Item Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((100, 100))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image
            img_array = preprocess_image(uploaded_image)

            # Make a prediction using the pre-trained model
            result = model.predict(img_array)
            # st.write(str(result))
            predicted_class = np.argmax(result)
            prediction = class_names[predicted_class]

            st.success(f'Prediction: {prediction}')