import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from config import IMG_SIZE

def predict_faces(model, face_paths):
    preds = []

    for p in face_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue

        img = cv2.resize(img, IMG_SIZE)
        img = preprocess_input(img.astype(np.float32))
        img = np.expand_dims(img, 0)

        prob = float(model.predict(img, verbose=0)[0][0])
        preds.append(prob)

    return preds

from config import FRAME_FAKE_THRESH, VIDEO_FAKE_RATIO, MIN_FRAMES_VALID

def aggregate_prediction(preds):
    if len(preds) < MIN_FRAMES_VALID:
        return "UNCERTAIN", 0.0

    preds = np.array(preds)

    fake_frames = preds > FRAME_FAKE_THRESH
    fake_ratio = fake_frames.mean()
    mean_score = preds.mean()

    if fake_ratio >= VIDEO_FAKE_RATIO:
        label = "FAKE"
    else:
        label = "REAL"

    return label, {
        "mean_score": round(mean_score, 4),
        "fake_ratio": round(float(fake_ratio), 4),
        "num_frames": len(preds),
    }
