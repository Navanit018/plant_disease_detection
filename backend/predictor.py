import numpy as np
from PIL import Image
import tensorflow as tf
import json
import os

# ===============================
# Load disease information
# ===============================
JSON_PATH = "plant_disease.json"

with open(JSON_PATH, "r") as f:
    disease_data = json.load(f)

CLASS_NAMES = [d["name"] for d in disease_data]

# ===============================
# Load trained model
# ===============================
MODEL_PATH = "plant_disease_model.h5"  # change if needed

model = tf.keras.models.load_model(MODEL_PATH)

# ===============================
# Image preprocessing
# ===============================
def preprocess(image):
    image = image.convert("RGB")
    image = image.resize((160, 160))
    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ===============================
# Prediction function
# ===============================
def predict(image, model):
    img = preprocess(image)
    pred = model.predict(img)[0]

    idx = int(np.argmax(pred))
    conf = float(pred[idx])

    disease = disease_data[idx]

    return {
        "Disease Name": disease["name"],
        "Confidence": round(conf * 100, 2),
        "Cause": disease.get("cause", "Not available"),
        "Cure": disease.get("cure", "Not available")
    }


