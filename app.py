import os
import cv2
import base64
import numpy as np
import insightface
from flask import Flask, render_template, request

# ---------------- CREATE FLASK APP FIRST ----------------
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------- LOAD ARCFACE MODEL ----------------
face_model = insightface.app.FaceAnalysis(name="buffalo_s")
face_model.prepare(ctx_id=0)

# ---------------- LOAD DATABASE ----------------
db_embeddings = []
db_names = []

print("Loading database...")

for file in os.listdir("faces_db"):
    path = os.path.join("faces_db", file)
    name = ''.join([c for c in file if not c.isdigit()]).split('.')[0]

    img = cv2.imread(path)
    faces = face_model.get(img)

    if faces:
        db_embeddings.append(faces[0].embedding)
        db_names.append(name)
        print(f"Loaded {name}")

print("Database Ready.")

# ---------------- SIMILARITY FUNCTION ----------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ---------------- ROUTES ----------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/match", methods=["POST"])
def match():
    image_data = request.form["imageData"]

    # Decode image from browser
    image_data = image_data.split(",")[1]
    image_bytes = base64.b64decode(image_data)

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], "capture.jpg")

    with open(file_path, "wb") as f:
        f.write(image_bytes)

    img = cv2.imread(file_path)
    faces = face_model.get(img)

    if not faces:
        return render_template("result.html", result="No face detected")

    emb = faces[0].embedding
    sims = [cosine_similarity(emb, db) for db in db_embeddings]
    best = np.argmax(sims)

    score = sims[best]

    if score > 0.4:
        result = f"Matched: {db_names[best]} (Confidence {score:.2f})"
    else:
        result = f"Unknown Person (Confidence {score:.2f})"

    return render_template("result.html", result=result)

# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)