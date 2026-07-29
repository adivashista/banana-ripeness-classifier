import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


features = np.load("features.npy")
labels = np.load("labels.npy")

print("Features Shape :", features.shape)
print("Labels Shape :", labels.shape)



encoder = LabelEncoder()

labels_encoded = encoder.fit_transform(labels)

print("\nClasses :", encoder.classes_)



X_train, X_test, y_train, y_test = train_test_split(
    features,
    labels_encoded,
    test_size=0.2,
    random_state=42,
    stratify=labels_encoded
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples :", len(X_test))



model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print("\nTraining Accuracy : {:.2f}%".format(train_accuracy * 100))
print("Testing Accuracy  : {:.2f}%".format(test_accuracy * 100))


joblib.dump(model, "rf_model.pkl")
joblib.dump(encoder, "label_encoder.pkl")

print("\n" + "=" * 50)
print("MODEL TRAINED SUCCESSFULLY")
print("=" * 50)
print("Model saved as : rf_model.pkl")
print("Label Encoder saved as : label_encoder.pkl")