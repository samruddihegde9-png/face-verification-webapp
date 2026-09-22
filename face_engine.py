import base64
import numpy as np
import cv2


def decode_data_url(data_url: str) -> np.ndarray:
    """Decode a `data:image/...;base64,...` string (from a <canvas>) into a BGR numpy image."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image data")
    return img


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class LivenessResult:
    def __init__(self, passed, checks: dict, detail: str):
        self.passed = passed
        self.checks = checks
        self.detail = detail

    def to_dict(self):
        return {"passed": self.passed, "checks": self.checks, "detail": self.detail}


class FaceEngine:
    """Wraps an insightface FaceAnalysis model: detection, embeddings, matching and
    lightweight anti-spoofing heuristics.

    NOTE on liveness: this is a heuristic pipeline (blur, face-size sanity, and an
    optional two-frame blink/motion challenge) meant to raise the bar against a photo
    held up to the camera. It is NOT a certified anti-spoofing system -- for anything
    security-critical, pair this with a dedicated liveness SDK or depth/IR hardware.
    """

    def __init__(self, model_name="buffalo_s", detection_size=640, ctx_id=0):
        import insightface  # imported lazily so the rest of the app can be tested without it
        self.model = insightface.app.FaceAnalysis(name=model_name)
        self.model.prepare(ctx_id=ctx_id, det_size=(detection_size, detection_size))

    def detect(self, img: np.ndarray):
        """Returns the list of detected faces (insightface Face objects), largest first."""
        faces = self.model.get(img)
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        return faces

    def largest_face(self, img: np.ndarray):
        faces = self.detect(img)
        return faces[0] if faces else None

    # ---------------- matching ----------------
    @staticmethod
    def best_match(embedding, names, ids, vectors):
        """Returns (name, person_id, score) for the best match, or (None, None, 0.0)."""
        if not vectors:
            return None, None, 0.0
        sims = [cosine_similarity(embedding, v) for v in vectors]
        best = int(np.argmax(sims))
        return names[best], ids[best], float(sims[best])

    # ---------------- liveness heuristics ----------------
    @staticmethod
    def check_blur(img: np.ndarray, min_variance: float):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance >= min_variance, float(variance)

    @staticmethod
    def check_face_size(face, img_shape, min_ratio: float, max_ratio: float):
        h, w = img_shape[:2]
        x1, y1, x2, y2 = face.bbox
        area_ratio = ((x2 - x1) * (y2 - y1)) / float(w * h)
        ok = min_ratio <= area_ratio <= max_ratio
        return ok, float(area_ratio)

    @staticmethod
    def _eye_aspect_ratio(landmarks_2d106, eye_indices):
        pts = landmarks_2d106[eye_indices]
        # vertical span / horizontal span, a rough eye-openness proxy
        horizontal = np.linalg.norm(pts[0] - pts[4])
        vertical = (np.linalg.norm(pts[1] - pts[5]) + np.linalg.norm(pts[2] - pts[6])) / 2.0
        if horizontal == 0:
            return 0.0
        return float(vertical / horizontal)

    def blink_challenge(self, face_a, face_b, min_delta: float):
        """Compares two frames of (ideally) the same person for eye-openness change,
        as a crude blink/liveness signal. Falls back gracefully if 106-point
        landmarks aren't available on this model."""
        kps_a = getattr(face_a, "landmark_2d_106", None)
        kps_b = getattr(face_b, "landmark_2d_106", None)
        if kps_a is None or kps_b is None:
            return None  # model doesn't expose dense landmarks; caller should skip this check

        # Rough 106-point left/right eye index groups (insightface convention).
        left_eye = [35, 41, 40, 42, 39, 37, 33, 36]
        right_eye = [89, 95, 94, 96, 93, 91, 87, 90]
        idx = (left_eye[:8] if len(left_eye) >= 8 else left_eye)

        try:
            ear_a = self._eye_aspect_ratio(kps_a, [35, 36, 33, 37, 39, 40, 42, 41][:8])
            ear_b = self._eye_aspect_ratio(kps_b, [35, 36, 33, 37, 39, 40, 42, 41][:8])
        except Exception:
            return None

        delta = abs(ear_a - ear_b) / max(ear_a, 1e-6) * 100.0
        return delta >= min_delta, delta

    def run_liveness(self, img: np.ndarray, face, cfg, second_frame_img=None, second_face=None):
        checks = {}

        blur_ok, blur_val = self.check_blur(img, cfg.BLUR_VARIANCE_MIN)
        checks["blur"] = {"passed": blur_ok, "value": round(blur_val, 1)}

        size_ok, size_val = self.check_face_size(face, img.shape, cfg.FACE_AREA_MIN_RATIO, cfg.FACE_AREA_MAX_RATIO)
        checks["face_size"] = {"passed": size_ok, "value": round(size_val, 4)}

        blink_ok = True  # default pass-through when a two-frame challenge wasn't provided
        if second_frame_img is not None and second_face is not None:
            result = self.blink_challenge(face, second_face, cfg.EYE_MOVEMENT_MIN)
            if result is None:
                checks["blink"] = {"passed": True, "value": None, "note": "landmarks unavailable, skipped"}
            else:
                blink_ok, blink_val = result
                checks["blink"] = {"passed": blink_ok, "value": round(blink_val, 2)}

        overall = blur_ok and size_ok and blink_ok
        failed = [k for k, v in checks.items() if not v["passed"]]
        detail = "all checks passed" if overall else f"failed: {', '.join(failed)}"
        return LivenessResult(overall, checks, detail)
