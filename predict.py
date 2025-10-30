# predict.py
import os
import numpy as np
import librosa
import pickle
from tensorflow.keras.models import load_model

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # → src/
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))    # → musicdata/

MODEL_PATH = os.path.join(PROJECT_DIR, "model", "music_genre_classifier.h5")
SCALER_PATH = os.path.join(PROJECT_DIR, "model", "scaler.pkl")
LABELS_PATH = os.path.join(PROJECT_DIR, "model", "label_classes.npy")

# Load model, scaler, and labels
model = load_model(MODEL_PATH)
scaler = pickle.load(open(SCALER_PATH, "rb"))
label_classes = np.load(LABELS_PATH, allow_pickle=True)

def extract_features(file_path):
    """Extract 58 features (same as training)."""
    y, sr = librosa.load(file_path, mono=True, duration=30)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    mfccs_mean = np.mean(mfccs, axis=1)
    chroma_mean = np.mean(chroma, axis=1)
    mel_mean = np.mean(mel, axis=1)
    contrast_mean = np.mean(contrast, axis=1)

    # Only 58 features (20 + 12 + 20 + 6)
    features = np.hstack([
        mfccs_mean[:20],
        chroma_mean[:12],
        mel_mean[:20],
        contrast_mean[:6]
    ])
    return features.reshape(1, -1)

def predict_genre(file_path):
    """Predict genre for a given audio file."""
    try:
        features = extract_features(file_path)
        features = scaler.transform(features)

        prediction = model.predict(features)
        predicted_label = label_classes[np.argmax(prediction)]
        return predicted_label
    except Exception as e:
        return f"❌ Error: {str(e)}"

if __name__ == "__main__":
    # Example: python predict.py data/genres_original/blues/blues.00000.wav
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <audio_file>")
    else:
        audio_file = sys.argv[1]
        if not os.path.exists(audio_file):
            print(f"❌ File not found: {audio_file}")
        else:
            genre = predict_genre(audio_file)
            print(f"🎵 Predicted Genre: {genre}")
