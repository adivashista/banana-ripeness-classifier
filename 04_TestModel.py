import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

features = np.load("features.npy")
labels = np.load("labels.npy")

encoder = joblib.load("label_encoder.pkl")
labels_encoded = encoder.transform(labels)

X_train, X_test, y_train, y_test = train_test_split(
    features,
    labels_encoded,
    test_size=0.2,
    random_state=42,
    stratify=labels_encoded
)

model = joblib.load("rf_model.pkl")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy : {accuracy * 100:.2f}%")

print("\nClassification Report\n")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, y_pred))

print("\nTop 10 Important Features\n")

importance = model.feature_importances_
top10 = np.argsort(importance)[::-1][:10]

for i in top10:
    print(f"Feature {i}: {importance[i]:.4f}")