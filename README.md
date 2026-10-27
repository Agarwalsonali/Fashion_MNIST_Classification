# Fashion MNIST Classification

A TensorFlow and Streamlit application that classifies uploaded clothing images into one of the ten Fashion MNIST categories.

## Features

- Upload a clothing image in JPG, JPEG, or PNG format.
- Convert the image to grayscale and resize it to the model's expected `28 x 28` input.
- Predict one of the ten Fashion MNIST classes using a pre-trained TensorFlow model.
- Display the uploaded image and the predicted class in a Streamlit web interface.

## Classes

The model predicts one of these labels:

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

## Project structure

```text
.
├── app/
│   ├── main.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── trained_model/
│       └── trained_fashion_mnist_model.h5
├── model_training_notebook/
│   └── Fashion_MNIST_model_training.ipynb
└── test_images/
    ├── fashion_mnist_1.png
    ├── fashion_mnist_2.jpg
    └── fashion_mnist_3.png
```

## Requirements

- Python 3.10 or later
- pip

The application dependencies are listed in [`app/requirements.txt`](app/requirements.txt):

- NumPy
- Streamlit
- TensorFlow
- h5py
- Pillow

## Run locally

From the repository root, create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the dependencies and start Streamlit:

```bash
pip install -r app/requirements.txt
streamlit run app/main.py
```

Open the URL shown by Streamlit, typically `http://localhost:8501`, then upload an image and select **Classify**.

## Run with Docker

Build the image using the `app` directory as the Docker build context:

```bash
docker build -t fashion-mnist-classifier ./app
```

Run the container:

```bash
docker run --rm -p 8501:80 fashion-mnist-classifier
```

Open `http://localhost:8501` in a browser.

## Training

The model-training notebook is available at [`model_training_notebook/Fashion_MNIST_model_training.ipynb`](model_training_notebook/Fashion_MNIST_model_training.ipynb). The trained model used by the Streamlit app is stored at [`app/trained_model/trained_fashion_mnist_model.h5`](app/trained_model/trained_fashion_mnist_model.h5).

## Sample images

The [`test_images/`](test_images/) directory contains sample images that can be uploaded to test the application.

## Notes

- The application normalizes grayscale pixel values to the range `0` to `1`.
- The included compatibility loader supports loading the bundled H5 model across relevant Keras model-format versions.
- The model is intended for Fashion MNIST-style images. Results may be less reliable for photographs or images with backgrounds and layouts that differ substantially from the training data.
