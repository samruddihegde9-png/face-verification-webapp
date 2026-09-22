from functools import wraps
from flask import Blueprint, request, jsonify, current_app

from face_engine import decode_data_url

api_bp = Blueprint("api", __name__, url_prefix="/api/v1")


def require_api_key(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        db = current_app.config["DB"]
        provided = request.headers.get("X-API-Key", "")
        expected = db.get_setting("api_key")
        if not expected or provided != expected:
            return jsonify({"error": "Missing or invalid X-API-Key header"}), 401
        return view(*args, **kwargs)
    return wrapped


@api_bp.route("/persons", methods=["GET"])
@require_api_key
def list_persons():
    db = current_app.config["DB"]
    persons = db.list_persons()
    return jsonify([
        {
            "id": p["id"],
            "name": p["name"],
            "embedding_count": p["embedding_count"],
            "created_at": p["created_at"],
        }
        for p in persons
    ])


@api_bp.route("/verify", methods=["POST"])
@require_api_key
def api_verify():
    """Single-frame verification for server-to-server use. Liveness challenge (which
    needs two frames from a live camera) is intentionally not enforced here by default
    -- pass "liveness": true with an "image2" if your client can supply a second frame."""
    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image")
    if not image_data:
        return jsonify({"error": "image is required (base64 data URL)"}), 400

    engine = current_app.config["FACE_ENGINE"]
    db = current_app.config["DB"]
    cfg = current_app.config["APP_CFG"]

    try:
        img = decode_data_url(image_data)
    except Exception:
        return jsonify({"error": "could not decode image"}), 400

    face = engine.largest_face(img)
    if face is None:
        db.add_log(person_id=None, matched_name=None, score=0.0, decision="no_face",
                    liveness_passed=None, liveness_detail=None, image_path=None,
                    source="api", ip_address=request.remote_addr)
        return jsonify({"decision": "no_face"})

    liveness_dict = None
    if payload.get("liveness"):
        img2 = None
        face2 = None
        if payload.get("image2"):
            try:
                img2 = decode_data_url(payload["image2"])
                face2 = engine.largest_face(img2)
            except Exception:
                pass
        liveness = engine.run_liveness(img, face, cfg, img2, face2)
        liveness_dict = liveness.to_dict()
        if not liveness.passed:
            db.add_log(person_id=None, matched_name=None, score=0.0, decision="liveness_fail",
                        liveness_passed=False, liveness_detail=liveness.detail, image_path=None,
                        source="api", ip_address=request.remote_addr)
            return jsonify({"decision": "liveness_fail", "liveness": liveness_dict})

    names, ids, vectors = db.all_embeddings()
    name, person_id, score = engine.best_match(face.embedding, names, ids, vectors)
    threshold = float(db.get_setting("match_threshold", cfg.MATCH_THRESHOLD_DEFAULT))
    margin = float(db.get_setting("review_margin", cfg.REVIEW_MARGIN))

    if name is None:
        decision = "unknown"
    elif score >= threshold:
        decision = "matched"
    elif score >= threshold - margin:
        decision = "review"
    else:
        decision = "unknown"

    db.add_log(person_id=person_id if decision == "matched" else None, matched_name=name,
               score=score, decision=decision, liveness_passed=liveness_dict["passed"] if liveness_dict else None,
               liveness_detail=liveness_dict["detail"] if liveness_dict else "not requested",
               image_path=None, source="api", ip_address=request.remote_addr)

    return jsonify({
        "decision": decision,
        "name": name,
        "score": round(score, 4),
        "threshold": threshold,
        "liveness": liveness_dict,
    })
