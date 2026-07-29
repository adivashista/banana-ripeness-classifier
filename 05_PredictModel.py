import cv2
import numpy as np
import joblib
from tkinter import Tk
from tkinter.filedialog import askopenfilename

Tk().withdraw()

image_path = askopenfilename(
    title="Select Banana Image",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
)

if not image_path:
    print("No image selected.")
    exit()

model = joblib.load("rf_model.pkl")
encoder = joblib.load("label_encoder.pkl")

image = cv2.imread(image_path)

if image is None:
    print("Unable to load image.")
    exit()

image = cv2.resize(image, (224, 224))

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
blur = cv2.GaussianBlur(hsv, (5, 5), 0)

lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([35, 255, 255])

lower_brown = np.array([5, 50, 20])
upper_brown = np.array([20, 255, 200])

mask_green = cv2.inRange(blur, lower_green, upper_green)
mask_yellow = cv2.inRange(blur, lower_yellow, upper_yellow)
mask_brown = cv2.inRange(blur, lower_brown, upper_brown)

mask = mask_green | mask_yellow | mask_brown

kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

banana_mask = np.zeros(mask.shape, dtype=np.uint8)

if len(contours) > 0:
    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(
        banana_mask,
        [largest],
        -1,
        255,
        -1
    )
else:
    banana_mask = mask

segmented = cv2.bitwise_and(
    image,
    image,
    mask=banana_mask
)

mean = cv2.mean(
    segmented,
    mask=banana_mask
)[:3]

std = cv2.meanStdDev(
    segmented,
    mask=banana_mask
)[1].flatten()

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

hist_b = cv2.normalize(hist_b, hist_b).flatten()
hist_g = cv2.normalize(hist_g, hist_g).flatten()
hist_r = cv2.normalize(hist_r, hist_r).flatten()

hist_h = cv2.normalize(hist_h, hist_h).flatten()
hist_s = cv2.normalize(hist_s, hist_s).flatten()
hist_v = cv2.normalize(hist_v, hist_v).flatten()

gray = cv2.cvtColor(
    segmented,
    cv2.COLOR_BGR2GRAY
)

edges = cv2.Canny(
    gray,
    100,
    200
)

edge_density = np.sum(edges > 0) / (224 * 224)

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

feature_vector = feature_vector.reshape(1, -1)

prediction = model.predict(feature_vector)

predicted_label = encoder.inverse_transform(prediction)[0]

probabilities = model.predict_proba(feature_vector)

confidence = np.max(probabilities) * 100

print("\nPrediction Result")
print("-------------------------")
print("Predicted Class :", predicted_label)
print(f"Confidence      : {confidence:.2f}%")