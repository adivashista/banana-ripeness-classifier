import json
import joblib
import numpy as np

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

FEATURES_PATH = "features.npy"
LABELS_PATH = "labels.npy"
MODEL_PATH = "rf_model.pkl"
ENCODER_PATH = "label_encoder.pkl"
METADATA_PATH = "model_metadata.json"

features = np.load(FEATURES_PATH)
labels = np.load(LABELS_PATH)

encoder = LabelEncoder()
y = encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(
    features,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

candidates = {
    "random_forest": RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        oob_score=True,
    ),
    "extra_trees": ExtraTreesClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}

scores = {}
for name, candidate in candidates.items():
    cv_scores = cross_val_score(candidate, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    scores[name] = {
        "cv_mean_accuracy": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
    }
    print(f"{name}: {cv_scores.mean() * 100:.2f}% +/- {cv_scores.std() * 100:.2f}%")

best_name = max(scores, key=lambda name: scores[name]["cv_mean_accuracy"])
best_model = candidates[best_name]
best_model.fit(X_train, y_train)

predictions = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, predictions)

print(f"\nSelected model: {best_name}")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")
print("\nClassification report:\n")
print(classification_report(y_test, predictions, target_names=encoder.classes_))

joblib.dump(best_model, MODEL_PATH)
joblib.dump(encoder, ENCODER_PATH)

metadata = {
    "model": best_name,
    "classes": encoder.classes_.tolist(),
    "test_accuracy": float(test_accuracy),
    "cross_validation": scores,
    "feature_count": int(features.shape[1]),
}
with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"\nSaved model to {MODEL_PATH}")
print(f"Saved encoder to {ENCODER_PATH}")
print(f"Saved metadata to {METADATA_PATH}")
