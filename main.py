from pathlib import Path

import cv2
import joblib
import numpy as np

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

FRONTEND_DIR = BASE_DIR / "frontend"
MODEL_PATH = BASE_DIR / "rf_model.pkl"
ENCODER_PATH = BASE_DIR / "label_encoder.pkl"


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="🍌 Banana Ripeness Classifier API",
    description="AI-powered banana ripeness classification",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# FRONTEND STATIC FILES
# ============================================================

if FRONTEND_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(FRONTEND_DIR)),
        name="static",
    )


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Model not found: {MODEL_PATH}\n"
        "Run 03_TrainModel.py first."
    )

if not ENCODER_PATH.exists():
    raise RuntimeError(
        f"Label encoder not found: {ENCODER_PATH}\n"
        "Run 03_TrainModel.py first."
    )


model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(image: np.ndarray) -> np.ndarray:

    # Resize
    image = cv2.resize(image, (224, 224))

    # --------------------------------------------------------
    # HSV
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    blur = cv2.GaussianBlur(
        hsv,
        (5, 5),
        0
    )

    # --------------------------------------------------------
    # COLOR RANGES
    # --------------------------------------------------------

    lower_green = np.array([
        35,
        40,
        40
    ])

    upper_green = np.array([
        85,
        255,
        255
    ])

    lower_yellow = np.array([
        20,
        100,
        100
    ])

    upper_yellow = np.array([
        35,
        255,
        255
    ])

    lower_brown = np.array([
        5,
        50,
        20
    ])

    upper_brown = np.array([
        20,
        255,
        200
    ])

    # --------------------------------------------------------
    # MASKS
    # --------------------------------------------------------

    mask_green = cv2.inRange(
        blur,
        lower_green,
        upper_green
    )

    mask_yellow = cv2.inRange(
        blur,
        lower_yellow,
        upper_yellow
    )

    mask_brown = cv2.inRange(
        blur,
        lower_brown,
        upper_brown
    )

    mask = (
        mask_green
        | mask_yellow
        | mask_brown
    )

    # --------------------------------------------------------
    # MORPHOLOGY
    # --------------------------------------------------------

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # --------------------------------------------------------
    # CONTOURS
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    banana_mask = np.zeros(
        mask.shape,
        dtype=np.uint8
    )

    if len(contours) > 0:

        largest = max(
            contours,
            key=cv2.contourArea
        )

        cv2.drawContours(
            banana_mask,
            [largest],
            -1,
            255,
            -1
        )

    else:

        banana_mask = mask

    # --------------------------------------------------------
    # SEGMENTATION
    # --------------------------------------------------------

    segmented = cv2.bitwise_and(
        image,
        image,
        mask=banana_mask
    )

    # --------------------------------------------------------
    # MEAN
    # --------------------------------------------------------

    mean = cv2.mean(
        segmented,
        mask=banana_mask
    )[:3]

    # --------------------------------------------------------
    # STANDARD DEVIATION
    # --------------------------------------------------------

    std = cv2.meanStdDev(
        segmented,
        mask=banana_mask
    )[1].flatten()

    # --------------------------------------------------------
    # BGR HISTOGRAMS
    # --------------------------------------------------------

    hist_b = cv2.calcHist(
        [segmented],
        [0],
        banana_mask,
        [16],
        [0, 256]
    )

    hist_g = cv2.calcHist(
        [segmented],
        [1],
        banana_mask,
        [16],
        [0, 256]
    )

    hist_r = cv2.calcHist(
        [segmented],
        [2],
        banana_mask,
        [16],
        [0, 256]
    )

    # --------------------------------------------------------
    # HSV HISTOGRAMS
    # --------------------------------------------------------

    hsv_image = cv2.cvtColor(
        segmented,
        cv2.COLOR_BGR2HSV
    )

    hist_h = cv2.calcHist(
        [hsv_image],
        [0],
        banana_mask,
        [16],
        [0, 180]
    )

    hist_s = cv2.calcHist(
        [hsv_image],
        [1],
        banana_mask,
        [16],
        [0, 256]
    )

    hist_v = cv2.calcHist(
        [hsv_image],
        [2],
        banana_mask,
        [16],
        [0, 256]
    )

    # --------------------------------------------------------
    # NORMALIZE HISTOGRAMS
    # --------------------------------------------------------

    hist_b = cv2.normalize(
        hist_b,
        hist_b
    ).flatten()

    hist_g = cv2.normalize(
        hist_g,
        hist_g
    ).flatten()

    hist_r = cv2.normalize(
        hist_r,
        hist_r
    ).flatten()

    hist_h = cv2.normalize(
        hist_h,
        hist_h
    ).flatten()

    hist_s = cv2.normalize(
        hist_s,
        hist_s
    ).flatten()

    hist_v = cv2.normalize(
        hist_v,
        hist_v
    ).flatten()

    # --------------------------------------------------------
    # EDGE DENSITY
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        segmented,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(
        gray,
        100,
        200
    )

    edge_density = (
        np.sum(edges > 0)
        / (224 * 224)
    )

    # --------------------------------------------------------
    # FINAL FEATURE VECTOR
    # --------------------------------------------------------

    feature_vector = np.hstack([
        mean,
        std,

        hist_b,
        hist_g,
        hist_r,

        hist_h,
        hist_s,
        hist_v,

        edge_density
    ])

    return feature_vector.reshape(
        1,
        -1
    )


# ============================================================
# HOME PAGE
# ============================================================

@app.get(
    "/",
    include_in_schema=False
)
def home():

    index_file = FRONTEND_DIR / "index.html"

    if not index_file.exists():

        return {
            "message": "Banana Ripeness Classifier API",
            "docs": "/docs",
            "error": "frontend/index.html not found"
        }

    return FileResponse(
        str(index_file)
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model_loaded": True,
        "classes": encoder.classes_.tolist()
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # VALIDATE FILE TYPE
    # --------------------------------------------------------

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/bmp"
    }

    if file.content_type not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image type. "
                "Please upload JPG, PNG, or BMP."
            )
        )

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    contents = await file.read()

    if not contents:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    # --------------------------------------------------------
    # DECODE IMAGE
    # --------------------------------------------------------

    image_array = np.frombuffer(
        contents,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise HTTPException(
            status_code=400,
            detail="Unable to decode uploaded image."
        )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    try:

        feature_vector = extract_features(
            image
        )

        prediction = model.predict(
            feature_vector
        )

        predicted_label = (
            encoder.inverse_transform(
                prediction
            )[0]
        )

        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        probabilities = model.predict_proba(
            feature_vector
        )[0]

        confidence = float(
            np.max(probabilities) * 100
        )

        class_probabilities = {}

        for label, probability in zip(
            encoder.classes_,
            probabilities
        ):

            class_probabilities[
                str(label)
            ] = round(
                float(probability * 100),
                2
            )

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return {
            "success": True,
            "filename": file.filename,
            "predicted_class": str(
                predicted_label
            ),
            "confidence": round(
                confidence,
                2
            ),
            "probabilities": class_probabilities
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}"
        )git status