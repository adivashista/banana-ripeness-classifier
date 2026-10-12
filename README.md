# 🍌 Banana Ripeness Classifier

A machine learning project that classifies banana images into four ripeness stages: **Raw, Ripe, Overripe, and Rotten**. The project uses handcrafted image features and tree-based ensemble classifiers, with a FastAPI application for image prediction.

## 📌 Project Overview

This project uses traditional computer vision techniques and machine learning to predict the ripeness stage of a banana from an image.

Instead of using a deep learning model, it extracts colour and edge features from images and trains two ensemble classifiers, Random Forest and Extra Trees. The models are compared using stratified 5-fold cross-validation, and the better-performing model is selected for evaluation.

### Features

- Four-class banana ripeness classification.
- Image preprocessing using OpenCV and NumPy.
- A fixed **103-dimensional feature vector** for every image.
- Random Forest and Extra Trees model comparison.
- Stratified train-test splitting and 5-fold cross-validation.
- Model evaluation using accuracy, precision, recall, F1-score and a confusion matrix.
- Image prediction through a Python script.
- FastAPI-based web service for uploading an image and receiving a prediction.
- Class-wise model confidence scores.

## 🏷️ Ripeness Classes

| Class | Description |
|---|---|
| Raw | Green or unripe banana |
| Ripe | Yellow banana suitable for consumption |
| Overripe | Banana with significant brown spots or darkening |
| Rotten | Banana showing extensive darkening or visible deterioration |

The model predicts visual ripeness from the image. It does not determine food safety, taste or remaining shelf life.

## 🧠 Feature Extraction

Each input image is resized and processed to generate a **103-feature vector**.

| Feature group | Number of features |
|---|---:|
| Mean colour values | 3 |
| Colour standard deviations | 3 |
| RGB histograms (16 bins per channel) | 48 |
| HSV histograms (16 bins per channel) | 48 |
| Edge density | 1 |
| **Total** | **103** |

These features describe the overall colour distribution and edge information in an image.

The same 103-feature representation is used for training, testing and prediction. This consistency is important because the trained model expects the same feature layout during inference.

## 📂 Project Structure

```text
banana-ripeness-classifier/
│
├── 01_CountImages.py
├── 02_PreProcessingFeatureExtraction.py
├── 03_TrainModel.py
├── 04_TestModel.py
├── 05_PredictModel.py
├── feature_pipeline.py
├── main.py
├── requirements.txt
├── README.md
│
├── features.npy
├── labels.npy
├── rf_model.pkl
└── label_encoder.pkl
```

**Main files**

- `01_CountImages.py`: Counts the images available in the dataset.
- `02_PreProcessingFeatureExtraction.py`: Extracts the 103 features and prepares the feature and label arrays.
- `03_TrainModel.py`: Trains and compares the candidate classifiers, evaluates their performance and saves the selected model.
- `04_TestModel.py`: Evaluates the saved model and produces classification metrics and a confusion matrix.
- `05_PredictModel.py`: Predicts the ripeness class of an individual image.
- `main.py`: Implements the FastAPI application for image-based prediction.
- `requirements.txt`: Lists the Python dependencies.
- `features.npy` and `labels.npy`: Store the extracted features and corresponding labels.
- `rf_model.pkl`: Stores the selected trained estimator.
- `label_encoder.pkl`: Stores the label encoder used for class labels.

`feature_pipeline.py` is present in the repository, but the reported results and documented workflow use the 103-feature implementation described above. The experimental feature pipeline is not used to claim the reported model performance.

## 🛠️ Technologies Used

- Python
- OpenCV
- NumPy
- scikit-learn
- FastAPI
- Uvicorn
- Joblib
- Matplotlib
- Seaborn

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/adivashista/banana-ripeness-classifier.git
cd banana-ripeness-classifier
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux/macOS**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 📊 Dataset Preparation

The project uses a banana ripeness image dataset containing four classes: Raw, Ripe, Overripe and Rotten.

Arrange the images in the directory structure expected by the preprocessing script. Check the dataset path and class-folder names in `01_CountImages.py` and `02_PreProcessingFeatureExtraction.py` before running the scripts.

Run the image-counting script:

```bash
python 01_CountImages.py
```

This helps verify the number of images available in each class before feature extraction.

## 🚀 Training the Model

Run the preprocessing script to extract the 103-dimensional feature vectors:

```bash
python 02_PreProcessingFeatureExtraction.py
```

The script generates the feature and label arrays used by the training process.

Train and compare the classifiers:

```bash
python 03_TrainModel.py
```

The training process compares Random Forest and Extra Trees using stratified 5-fold cross-validation. The selected estimator and label encoder are saved for subsequent prediction.

**Important:** Training and inference must use the same 103-feature extraction procedure. Do not substitute a different feature layout without retraining and evaluating the model.

## 📈 Model Evaluation

Run the testing script:

```bash
python 04_TestModel.py
```

The evaluation includes:

- Accuracy
- Per-class precision, recall and F1-score
- Macro and weighted averages
- Confusion matrix
- Feature importance

### Reported Results

The reported evaluation of the 103-feature pipeline produced the following results:

| Metric | Result |
|---|---:|
| Dataset size | 11,793 images |
| Training set | 9,434 images |
| Test set | 2,359 images |
| Random Forest cross-validation accuracy | 89.98% ± 1.06 |
| Extra Trees cross-validation accuracy | 90.75% ± 0.97 |
| Extra Trees test accuracy | 91.01% |
| Macro F1-score | 0.92 |

These values describe the reported experiment on the existing dataset split. They should not be interpreted as guaranteed accuracy on images captured under different real-world conditions.

## 🔍 Predicting a Banana's Ripeness

To predict an individual image, use the prediction script:

```bash
python 05_PredictModel.py
```

Follow the input instructions implemented in the script to provide an image path.

The model returns one of the four ripeness classes. Make sure the trained model files are available and that the prediction script uses the same 103-feature extraction procedure used during training.

## 🌐 Running the FastAPI Application

Start the API server from the project root:

```bash
uvicorn main:app --reload
```

If Uvicorn is not installed, install it using:

```bash
pip install uvicorn
```

Once the server starts, open:

```text
http://127.0.0.1:8000/docs
```

The FastAPI documentation page allows you to inspect the available endpoints and test the API.

The application accepts an image for prediction and returns the predicted ripeness class along with class-wise model confidence scores.

The application must be able to access the trained model and label encoder files before predictions can be made.

## ⚠️ Limitations

- The model uses handcrafted global colour and edge features rather than learned image representations.
- Backgrounds, shadows and lighting conditions can affect predictions.
- The classifier is intended for images of individual bananas.
- Adjacent ripeness classes can have similar visual characteristics.
- The reported test results come from the same dataset source used for training.
- Model confidence scores represent tree-vote proportions and are not necessarily calibrated probabilities.
- The model does not estimate remaining shelf life or determine whether a banana is safe to eat.

## 🔮 Future Improvements

Potential future work includes:

- Evaluating the model on an independent real-world image dataset.
- Improving robustness to lighting and background variation.
- Investigating image segmentation and richer texture features in a separate experiment.
- Comparing additional classifiers and calibrated probability estimates.
- Exploring deep learning and transfer learning.
- Developing a more polished web or mobile interface.

Any alternative feature-extraction pipeline should be evaluated separately and should not be confused with the 103-feature model results reported here.

## 👨‍💻 Author

**Aditya Vashista**

GitHub: [adivashista](https://github.com/adivashista)

