import os
import base64
import uuid
import cv2
from flask import Blueprint, render_template, request, jsonify, current_app

from face_engine import decode_data_url

main_bp = Blueprint("main", __name__)


def _save_capture(img, prefix="capture"):
    uploads = current_app.config["UPLOAD_FOLDER"]
    os.makedirs(uploads, exist_ok=True)
    filename = f"{prefix}_{uuid.uuid4().hex[:10]}.jpg"
    path = os.path.join(uploads, filename)
    cv2.imwrite(path, img)
    return path


def _annotate(img, face, color=(0, 200, 0), label=None):
    out = img.copy()
    if face is not None:
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if label:
            cv2.putText(out, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    ok, buf = cv2.imencode(".jpg", out)
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii") if ok else None


@main_bp.route("/")
def home():
    return render_template("index.html")


@main_bp.route("/verify", methods=["POST"])
def verify():
    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image")
    image2_data = payload.get("image2")
    liveness_requested = bool(payload.get("liveness", True))

    if not image_data:
        return jsonify({"decision": "error", "message": "No image provided"}), 400

    engine = current_app.config["FACE_ENGINE"]
    db = current_app.config["DB"]

    try:
        img = decode_data_url(image_data)
    except Exception:
        return jsonify({"decision": "error", "message": "Could not decode image"}), 400

    face = engine.largest_face(img)
    if face is None:
        _log(db, None, None, 0.0, "no_face", None, "no face detected", img, "web")
        return jsonify({"decision": "no_face", "message": "No face detected. Move closer / improve lighting."})

    img2, face2 = None, None
    if image2_data:
        try:
            img2 = decode_data_url(image2_data)
            face2 = engine.largest_face(img2)
        except Exception:
            img2, face2 = None, None

    require_liveness = liveness_requested and db.get_setting("require_liveness", "1") == "1"
    liveness = None
    if require_liveness:
        liveness = engine.run_liveness(img, face, current_app.config["APP_CFG"], img2, face2)
        if not liveness.passed:
            path = _save_capture(img, "liveness_fail")
            annotated = _annotate(img, face, color=(0, 0, 220), label="LIVENESS FAIL")
            _log(db, None, None, 0.0, "liveness_fail", False, liveness.detail, None, "web", path)
            return jsonify({
                "decision": "liveness_fail",
                "message": f"Liveness check failed ({liveness.detail}). Try again in better lighting, "
                           f"without a photo/screen in front of the camera.",
                "liveness": liveness.to_dict(),
                "annotated_image": annotated,
            })

    names, ids, vectors = db.all_embeddings()
    name, person_id, score = engine.best_match(face.embedding, names, ids, vectors)

    threshold = float(db.get_setting("match_threshold", current_app.config["APP_CFG"].MATCH_THRESHOLD_DEFAULT))
    margin = float(db.get_setting("review_margin", current_app.config["APP_CFG"].REVIEW_MARGIN))

    if name is None:
        decision, label, color = "unknown", "NO DATABASE", (0, 0, 220)
    elif score >= threshold:
        decision, label, color = "matched", f"{name} ({score:.2f})", (0, 200, 0)
    elif score >= threshold - margin:
        decision, label, color = "review", f"{name}? ({score:.2f})", (0, 165, 255)
    else:
        decision, label, color = "unknown", f"Unknown ({score:.2f})", (0, 0, 220)

    path = _save_capture(img, decision)
    annotated = _annotate(img, face, color=color, label=label)
    _log(db, person_id if decision == "matched" else None, name, score, decision,
         liveness.passed if liveness else None, liveness.detail if liveness else "not required",
         None, "web", path)

    return jsonify({
        "decision": decision,
        "name": name,
        "score": round(score, 4),
        "threshold": threshold,
        "liveness": liveness.to_dict() if liveness else None,
        "annotated_image": annotated,
    })


def _log(db, person_id, matched_name, score, decision, liveness_passed, liveness_detail,
         img, source, image_path=None):
    if image_path is None and img is not None:
        image_path = _save_capture(img, decision)
    db.add_log(
        person_id=person_id, matched_name=matched_name, score=score, decision=decision,
        liveness_passed=liveness_passed, liveness_detail=liveness_detail,
        image_path=image_path, source=source,
        ip_address=request.headers.get("X-Forwarded-For", request.remote_addr),
    )
