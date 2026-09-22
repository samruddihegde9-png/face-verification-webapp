import os
from flask import Flask

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from config import Config
from database import DB, bootstrap_defaults
from face_engine import FaceEngine


def create_app(config_class=Config):
    cfg = config_class()
    app = Flask(__name__)
    app.config.from_object(cfg)
    app.config["APP_CFG"] = cfg

    os.makedirs(cfg.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(os.path.dirname(cfg.DB_PATH), exist_ok=True)

    db = DB(cfg.DB_PATH)
    bootstrap_defaults(db, cfg)
    app.config["DB"] = db

    print(f"Loading face model '{cfg.MODEL_NAME}' ...")
    engine = FaceEngine(model_name=cfg.MODEL_NAME, detection_size=cfg.DETECTION_SIZE)
    app.config["FACE_ENGINE"] = engine
    print("Model ready.")

    from main_routes import main_bp
    from enroll_routes import enroll_bp
    from admin_routes import admin_bp
    from auth import auth_bp
    from api_routes import api_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(enroll_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(api_bp)

    @app.context_processor
    def inject_globals():
        return {"app_name": "Face Verification"}

    return app


if __name__ == "__main__":
    application = create_app()
    application.run(host="0.0.0.0", port=5000, debug=os.environ.get("FLASK_DEBUG", "0") == "1")
