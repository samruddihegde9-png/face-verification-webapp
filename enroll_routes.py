import re
from flask import Blueprint, render_template, request, jsonify, current_app

from face_engine import decode_data_url
from main_routes import _save_capture

enroll_bp = Blueprint("enroll", __name__)

NAME_RE = re.compile(r"^[A-Za-z0-9 _.\-]{2,60}$")


@enroll_bp.route("/enroll")
def enroll_page():
    return render_template("enroll.html")


@enroll_bp.route("/enroll", methods=["POST"])
def enroll_submit():
    payload = request.get_json(silent=True) or {}
    name = (payload.get("name") or "").strip()
    images = payload.get("images") or []

    if not NAME_RE.match(name):
        return jsonify({"ok": False, "message": "Name must be 2-60 characters (letters, numbers, spaces, - _ .)."}), 400
    if not images:
        return jsonify({"ok": False, "message": "Capture at least one photo."}), 400
    if len(images) > 6:
        images = images[:6]

    engine = current_app.config["FACE_ENGINE"]
    db = current_app.config["DB"]

    saved = 0
    skipped = []
    for i, data_url in enumerate(images):
        try:
            img = decode_data_url(data_url)
        except Exception:
            skipped.append(f"shot {i+1}: could not decode image")
            continue

        face = engine.largest_face(img)
        if face is None:
            skipped.append(f"shot {i+1}: no face detected")
            continue

        blur_ok, _ = engine.check_blur(img, current_app.config["APP_CFG"].BLUR_VARIANCE_MIN)
        if not blur_ok:
            skipped.append(f"shot {i+1}: image too blurry")
            continue

        person = db.get_person_by_name(name)
        person_id = person["id"] if person else db.create_person(name)

        path = _save_capture(img, f"enroll_{name}")
        db.add_embedding(person_id, face.embedding, image_path=path)
        saved += 1

    if saved == 0:
        return jsonify({"ok": False, "message": "No usable photos. " + "; ".join(skipped)}), 400

    message = f"Enrolled {saved} photo(s) for '{name}'."
    if skipped:
        message += " Skipped: " + "; ".join(skipped)
    return jsonify({"ok": True, "message": message, "saved": saved})
