
# 🌿 Plant Disease Detection System

A deep learning–based **Plant Disease Detection** project using **TensorFlow**, **Python**, and **Image Processing**.  
The system predicts plant diseases from leaf images and provides **disease name, confidence, cause, and cure**.

---

## 📂 Project Structure
plant-disease-detection/
│
├── backend/
│ ├── pycache/
│ └── predictor.py
│
├── training/
│ └── (model training scripts & notebooks)
│
├── uploadimages/
│ └── (uploaded leaf images
│
├── app.py
├── test.py
├── convert_to_tflite.py
├── plant_disease.json
├── requirements.txt
├── runtime.txt
└── README.md

---

## 🚀 Features

- 🌱 Plant leaf disease classification  
- 📊 Confidence score for prediction  
- 🦠 Disease cause identification  
- 💊 Cure & treatment suggestions  
- 🔄 TensorFlow → TFLite conversion support  
- 🖼 Image preprocessing using PIL  

---

## 🧠 Technologies Used

- Python 3.9+
- TensorFlow / Keras
- NumPy
- Pillow (PIL)
- JSON
- Streamlit / CLI (depending on usage)

---

## 📁 File Descriptions

### `app.py`
Main application file to run disease prediction using trained model.

### `backend/predictor.py`
Contains:
- Image preprocessing
- Model inference logic
- Disease metadata mapping

### `training/`
Contains training scripts and notebooks for building the CNN model.

### `uploadimages/`
Stores images uploaded for prediction.

### `plant_disease.json`
Metadata file containing:
- Disease name
- Cause
- Cure

### `convert_to_tflite.py`
Converts trained TensorFlow model (`.h5`) to TFLite format.

### `test.py`
Used for testing model predictions locally.

### `requirements.txt`
List of Python dependencies.

### `runtime.txt`
Specifies Python runtime version (useful for deployment).

---


