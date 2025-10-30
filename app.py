from flask import Flask, request, render_template, redirect, url_for
import os
import numpy as np
import librosa
import joblib
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../model/music_genre_classifier.h5")
SCALER_PATH = os.path.join(BASE_DIR, "../model/scaler.pkl")
CLASSES_PATH = os.path.join(BASE_DIR, "../model/label_classes.npy")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

# Create upload folder if not exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load model, scaler, and classes
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_classes = np.load(CLASSES_PATH, allow_pickle=True)

# Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050, duration=30)

    # --- MFCCs (20) ---
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfccs_mean = np.mean(mfccs, axis=1)

    # --- Chroma (12) ---
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # --- Mel (20) ---
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_mean = np.mean(mel, axis=1)

    # --- Contrast (6) ---
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)

    # Combine → 58 features
    features = np.hstack([
        mfccs_mean[:20],
        chroma_mean[:12],
        mel_mean[:20],
        contrast_mean[:6]
    ])

    return features


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Extract features
            features = extract_features(filepath).reshape(1, -1)
            features = scaler.transform(features)

            # Predict
            prediction = model.predict(features)
            predicted_label = label_classes[np.argmax(prediction)]

            return render_template("index.html", prediction=predicted_label)

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
    
