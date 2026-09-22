from functools import wraps
from flask import Blueprint, render_template, request, redirect, url_for, session, current_app
from werkzeug.security import check_password_hash

auth_bp = Blueprint("auth", __name__)


def require_admin(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("is_admin"):
            return redirect(url_for("auth.login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


@auth_bp.route("/admin/login", methods=["GET", "POST"])
def login():
    db = current_app.config["DB"]
    error = None
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        stored_user = db.get_setting("admin_username", "admin")
        stored_hash = db.get_setting("admin_password_hash")
        if username == stored_user and stored_hash and check_password_hash(stored_hash, password):
            session.clear()
            session["is_admin"] = True
            session["admin_user"] = username
            next_url = request.args.get("next") or url_for("admin.dashboard")
            return redirect(next_url)
        error = "Invalid username or password."
    return render_template("login.html", error=error)


@auth_bp.route("/admin/logout")
def logout():
    session.clear()
    return redirect(url_for("auth.login"))
