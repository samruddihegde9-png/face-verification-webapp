import os
import secrets

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def _bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


class Config:
    # Flask
    SECRET_KEY = os.environ.get("SECRET_KEY") or secrets.token_hex(32)

    # Storage
    DB_PATH = os.environ.get("DB_PATH", os.path.join(BASE_DIR, "instance", "app.db"))
    UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", os.path.join(BASE_DIR, "uploads"))
    LEGACY_FACES_DB = os.environ.get("LEGACY_FACES_DB", os.path.join(BASE_DIR, "faces_db"))

    # Face model
    MODEL_NAME = os.environ.get("MODEL_NAME", "buffalo_s")
    DETECTION_SIZE = int(os.environ.get("DETECTION_SIZE", "640"))

    # Matching
    MATCH_THRESHOLD_DEFAULT = float(os.environ.get("MATCH_THRESHOLD", "0.40"))
    REVIEW_MARGIN = float(os.environ.get("REVIEW_MARGIN", "0.07"))  # score within threshold-margin..threshold => "review"

    # Liveness heuristics
    BLUR_VARIANCE_MIN = float(os.environ.get("BLUR_VARIANCE_MIN", "60.0"))
    FACE_AREA_MIN_RATIO = float(os.environ.get("FACE_AREA_MIN_RATIO", "0.02"))
    FACE_AREA_MAX_RATIO = float(os.environ.get("FACE_AREA_MAX_RATIO", "0.85"))
    EYE_MOVEMENT_MIN = float(os.environ.get("EYE_MOVEMENT_MIN", "1.5"))  # min relative EAR delta for blink challenge
    REQUIRE_LIVENESS_DEFAULT = _bool(os.environ.get("REQUIRE_LIVENESS_DEFAULT"), True)

    # Admin auth
    ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
    ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD")  # if unset, a random one is generated & printed on first run

    # API
    API_KEY = os.environ.get("API_KEY")  # if unset, a random one is generated & stored in settings table

    # Misc
    MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB request cap
