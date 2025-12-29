import cv2
import numpy as np
from pathlib import Path
from face_extractor import FaceExtractor
from config import *

def sample_frames(cap, n):
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return []
    return np.linspace(0, total - 1, n, dtype=int)

def extract_faces_from_video(video_path):
    extractor = FaceExtractor()
    cap = cv2.VideoCapture(str(video_path))

    video_face_dir = FACE_DIR / video_path.stem
    video_face_dir.mkdir(parents=True, exist_ok=True)

    faces = []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return []

    num_frames = min(FRAMES_PER_VIDEO, total)
    frame_ids = sample_frames(cap, num_frames)

    for idx, fid in enumerate(frame_ids):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face, conf = extractor.detect(rgb)

        if face is None or conf < FACE_CONF_THRESH:
            continue

        face_path = video_face_dir / f"face_{idx:03d}.jpg"
        cv2.imwrite(str(face_path), cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
        faces.append(face_path)

    cap.release()
    return faces
