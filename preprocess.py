# preprocess.py
import os
import json
import librosa
import numpy as np

DATASET_PATH = "data/genres_original"
OUTPUT_PATH = "data/data.json"
SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def extract_features(signal, sr):
    """Extract 58 audio features: MFCCs(20), Chroma(12), Mel(20), Contrast(6)."""
    try:
        # --- MFCCs (20) ---
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=20)
        mfccs_mean = np.mean(mfccs, axis=1)

        # --- Chroma (12) ---
        chroma = librosa.feature.chroma_stft(y=signal, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        # --- Mel Spectrogram (20) ---
        mel = librosa.feature.melspectrogram(y=signal, sr=sr)
        mel_mean = np.mean(mel, axis=1)

        # --- Spectral Contrast (6) ---
        contrast = librosa.feature.spectral_contrast(y=signal, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)

        # Combine → 58 features
        features = np.hstack([
            mfccs_mean[:20],
            chroma_mean[:12],
            mel_mean[:20],
            contrast_mean[:6]
        ])

        return features.tolist()
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        return None


def process_dataset(dataset_path, json_path):
    data = {
        "mapping": [],
        "features": [],
        "labels": []
    }

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath != dataset_path:  # not the root folder
            genre = os.path.split(dirpath)[-1]
            data["mapping"].append(genre)
            print(f"🎶 Processing {genre}...")

            for f in filenames:
                file_path = os.path.join(dirpath, f)
                try:
                    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                    if len(signal) >= SAMPLES_PER_TRACK:
                        signal = signal[:SAMPLES_PER_TRACK]

                        features = extract_features(signal, sr)
                        if features:
                            data["features"].append(features)
                            data["labels"].append(i - 1)  # because i starts from 1
                except Exception as e:
                    print(f"⚠️ Skipping {file_path} due to error: {e}")

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

    print(f"✅ Dataset processed and saved to {json_path}")


if __name__ == "__main__":
    process_dataset(DATASET_PATH, OUTPUT_PATH)
