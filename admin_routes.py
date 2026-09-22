import os
import secrets
from flask import Blueprint, render_template, request, redirect, url_for, current_app, flash, send_file, abort
from werkzeug.security import generate_password_hash, check_password_hash

from auth import require_admin

admin_bp = Blueprint("admin", __name__)


@admin_bp.route("/media")
@require_admin
def media():
    """Serves a captured/enrolled image by path, restricted to the configured
    upload folder to prevent path traversal."""
    requested = request.args.get("path", "")
    upload_root = os.path.realpath(current_app.config["UPLOAD_FOLDER"])
    legacy_root = os.path.realpath(current_app.config["APP_CFG"].LEGACY_FACES_DB)
    resolved = os.path.realpath(requested)
    if not (resolved.startswith(upload_root) or resolved.startswith(legacy_root)):
        abort(403)
    if not os.path.isfile(resolved):
        abort(404)
    return send_file(resolved)


@admin_bp.route("/admin")
@require_admin
def dashboard():
    db = current_app.config["DB"]
    persons = db.list_persons()
    stats = db.stats()
    return render_template("admin_dashboard.html", persons=persons, stats=stats)


@admin_bp.route("/admin/persons/<int:person_id>/delete", methods=["POST"])
@require_admin
def delete_person(person_id):
    db = current_app.config["DB"]
    db.delete_person(person_id)
    flash("Person deleted.", "success")
    return redirect(url_for("admin.dashboard"))


@admin_bp.route("/admin/persons/<int:person_id>")
@require_admin
def person_detail(person_id):
    db = current_app.config["DB"]
    person = db.get_person(person_id)
    embeddings = db.embeddings_for_person(person_id)
    logs = db.recent_logs(limit=50, person_id=person_id)
    return render_template("admin_person.html", person=person, embeddings=embeddings, logs=logs)


@admin_bp.route("/admin/embeddings/<int:embedding_id>/delete", methods=["POST"])
@require_admin
def delete_embedding(embedding_id):
    db = current_app.config["DB"]
    person_id = request.form.get("person_id", type=int)
    db.delete_embedding(embedding_id)
    flash("Photo removed.", "success")
    if person_id:
        return redirect(url_for("admin.person_detail", person_id=person_id))
    return redirect(url_for("admin.dashboard"))


@admin_bp.route("/admin/history")
@require_admin
def history():
    db = current_app.config["DB"]
    logs = db.recent_logs(limit=200)
    return render_template("admin_history.html", logs=logs)


@admin_bp.route("/admin/settings", methods=["GET", "POST"])
@require_admin
def settings():
    db = current_app.config["DB"]

    if request.method == "POST":
        action = request.form.get("action")

        if action == "update_matching":
            try:
                threshold = float(request.form.get("match_threshold"))
                margin = float(request.form.get("review_margin"))
            except (TypeError, ValueError):
                flash("Threshold and margin must be numbers.", "error")
                return redirect(url_for("admin.settings"))
            db.set_setting("match_threshold", threshold)
            db.set_setting("review_margin", margin)
            db.set_setting("require_liveness", "1" if request.form.get("require_liveness") else "0")
            flash("Matching settings updated.", "success")

        elif action == "regenerate_api_key":
            new_key = secrets.token_urlsafe(24)
            db.set_setting("api_key", new_key)
            flash(f"New API key generated: {new_key}", "success")

        elif action == "change_password":
            current_pw = request.form.get("current_password", "")
            new_pw = request.form.get("new_password", "")
            stored_hash = db.get_setting("admin_password_hash")
            if not stored_hash or not check_password_hash(stored_hash, current_pw):
                flash("Current password is incorrect.", "error")
            elif len(new_pw) < 8:
                flash("New password must be at least 8 characters.", "error")
            else:
                db.set_setting("admin_password_hash", generate_password_hash(new_pw))
                flash("Password updated.", "success")

        return redirect(url_for("admin.settings"))

    context = {
        "match_threshold": db.get_setting("match_threshold"),
        "review_margin": db.get_setting("review_margin"),
        "require_liveness": db.get_setting("require_liveness", "1") == "1",
        "api_key": db.get_setting("api_key"),
        "admin_username": db.get_setting("admin_username", "admin"),
    }
    return render_template("admin_settings.html", **context)
