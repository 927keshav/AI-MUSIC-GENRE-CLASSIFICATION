# src/predict.py
import os
import librosa
import numpy as np
import joblib
from tensorflow.keras.models import load_model

MODEL_PATH = "../model/music_genre_classifier.h5"
SCALER_PATH = "../model/scaler.pkl"
CLASSES_PATH = "../model/label_classes.npy"

# Load model + scaler + classes
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)   # ✅ FIXED: use joblib
label_classes = np.load(CLASSES_PATH, allow_pickle=True)

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    
    # same order as training (20 + 12 + 20 + 6 = 58 features)
    features = np.hstack([mfcc, chroma, mel, contrast])
    return features

if __name__ == "__main__":
    import sys
    file_path = sys.argv[1]
    features = extract_features(file_path).reshape(1, -1)
    features = scaler.transform(features)   # scale
    prediction = model.predict(features)
    predicted_label = label_classes[np.argmax(prediction)]
    print("Predicted Genre:", predicted_label)
