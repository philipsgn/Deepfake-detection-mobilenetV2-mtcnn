import cv2
from mtcnn import MTCNN
from config import *

class FaceExtractor:
    def __init__(self):
        self.detector = MTCNN()

    def _valid_box(self, box):
        x, y, w, h = box
        if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
            return False
        if w > MAX_FACE_SIZE or h > MAX_FACE_SIZE:
            return False
        ar = w / h if h > 0 else 0
        return 0.75 <= ar <= 1.33

    def crop_face(self, img, box):
        h, w, _ = img.shape
        x, y, bw, bh = box
        if not self._valid_box(box):
            return None

        mx, my = int(bw * MARGIN), int(bh * MARGIN)
        x1, y1 = max(0, x - mx), max(0, y - my)
        x2, y2 = min(w, x + bw + mx), min(h, y + bh + my)

        face = img[y1:y2, x1:x2]
        if face.size == 0:
            return None

        return cv2.resize(face, IMG_SIZE)

    def detect(self, frame_rgb):
        faces = self.detector.detect_faces(frame_rgb)
        if not faces:
            return None, 0.0

        valid_faces = [
        f for f in faces
        if self._valid_box(f["box"]) and f["confidence"] >= CONF_THRESH]

        if not valid_faces:
            return None, 0.0

        best = max(valid_faces, key=lambda x: x["confidence"])

        face = self.crop_face(frame_rgb, best["box"])
        return face, best["confidence"]
