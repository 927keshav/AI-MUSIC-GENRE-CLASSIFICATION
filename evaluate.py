# evaluate.py
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
import joblib

# --- Paths ---
DATA_PATH = "data/data.json"
MODEL_PATH = "model/music_genre_classifier.h5"
LABELS_PATH = "model/label_classes.npy"
SCALER_PATH = "model/scaler.pkl"

# --- Load model, labels, and scaler ---
print("🔄 Loading model & preprocessing files...")
model = tf.keras.models.load_model(MODEL_PATH)
label_classes = np.load(LABELS_PATH, allow_pickle=True)
scaler = joblib.load(SCALER_PATH)

# --- Load dataset ---
print("🔄 Loading dataset...")
with open(DATA_PATH, "r") as fp:
    data = json.load(fp)

X = np.array(data["features"])
y = np.array(data["labels"])

# --- Scale features ---
X_scaled = scaler.transform(X)

# --- Predict ---
print("🔮 Running predictions...")
y_pred = model.predict(X_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)

# --- Evaluate ---
print("\n✅ Accuracy:", accuracy_score(y, y_pred_classes))
print("\n📊 Classification Report:\n", classification_report(y, y_pred_classes, target_names=label_classes))
