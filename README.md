# 🍌 Banana Ripeness Classification using Machine Learning

## Overview

This project classifies bananas into four ripeness stages using classical Machine Learning with a Random Forest classifier.

The system preprocesses banana images, extracts color and texture features, trains a Random Forest model, and predicts the ripeness of new banana images through both a Python script and a Streamlit web application.

---

## Classes

- Raw
- Ripe
- Overripe
- Rotten

---

## Technologies Used

- Python
- OpenCV
- NumPy
- Scikit-learn
- Streamlit
- Pillow
- Joblib

---

## Project Structure

```
banana_classification/
│
├── train/
├── test/
├── valid/
│
├── 01_CountImages.py
├── 02_PreProcessing.py
├── 03_TrainModel.py
├── 04_TestModel.py
├── 05_PredictModel.py
├── 06_Streamlit.py
│
├── features.npy
├── labels.npy
├── rf_model.pkl
├── label_encoder.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Workflow

1. Count images in the dataset.
2. Preprocess banana images.
3. Extract 103 numerical features.
4. Train a Random Forest classifier.
5. Test the trained model.
6. Predict banana ripeness for new images.
7. Deploy the model using Streamlit.

---

## Feature Extraction

The following features are extracted from each banana image:

- Mean Color (3)
- Standard Deviation (3)
- RGB Histogram (48)
- HSV Histogram (48)
- Edge Density (1)

Total Features: **103**

---

## Machine Learning Model

Algorithm:

- Random Forest Classifier

---

## How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 1

```bash
python 01_CountImages.py
```

### Step 2

```bash
python 02_PreProcessing.py
```

### Step 3

```bash
python 03_TrainModel.py
```

### Step 4

```bash
python 04_TestModel.py
```

### Step 5

```bash
python 05_PredictModel.py
```

### Step 6

```bash
streamlit run 06_Streamlit.py
```

---

## Dataset

Dataset contains four banana ripeness classes:

- Raw
- Ripe
- Overripe
- Rotten

---

## Author

Aditya Vashista