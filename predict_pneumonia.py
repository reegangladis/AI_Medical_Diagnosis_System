import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load model
MODEL_PATH = "models/pneumonia_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Image settings
IMG_SIZE = (224, 224)

# Test image path
IMG_PATH = "test_images/test1.jpeg"   # change name if needed

# Load & preprocess image
img = image.load_img(IMG_PATH, target_size=IMG_SIZE)
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)[0][0]

# Output
if prediction > 0.5:
    print(f"Prediction: PNEUMONIA ({prediction*100:.2f}% confidence)")
else:
    print(f"Prediction: NORMAL ({(1-prediction)*100:.2f}% confidence)")
