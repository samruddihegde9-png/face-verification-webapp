"""
One-time migration: import the old flat-file `faces_db/*.jpg` photos (named like
`arj1.jpg`, `sinchana.jpg`) into the new SQLite-backed database.

Usage:
    python seed_from_faces_db.py
"""
import os
import re
import cv2

from config import Config
from database import DB, bootstrap_defaults
from face_engine import FaceEngine


def name_from_filename(filename):
    stem = os.path.splitext(filename)[0]
    # strip trailing digits, e.g. "arj1" -> "arj"
    stem = re.sub(r"\d+$", "", stem)
    return stem.strip().replace("_", " ").title() or os.path.splitext(filename)[0]


def main():
    cfg = Config()
    db = DB(cfg.DB_PATH)
    bootstrap_defaults(db, cfg)

    folder = cfg.LEGACY_FACES_DB
    if not os.path.isdir(folder):
        print(f"No legacy faces_db folder found at {folder}, nothing to do.")
        return

    print(f"Loading face model '{cfg.MODEL_NAME}' ...")
    engine = FaceEngine(model_name=cfg.MODEL_NAME, detection_size=cfg.DETECTION_SIZE)

    imported, skipped = 0, []
    for filename in sorted(os.listdir(folder)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(folder, filename)
        img = cv2.imread(path)
        if img is None:
            skipped.append(f"{filename}: could not read image")
            continue

        face = engine.largest_face(img)
        if face is None:
            skipped.append(f"{filename}: no face detected")
            continue

        name = name_from_filename(filename)
        person = db.get_person_by_name(name)
        person_id = person["id"] if person else db.create_person(name, notes="imported from faces_db")
        db.add_embedding(person_id, face.embedding, image_path=path)
        imported += 1
        print(f"  imported {filename} -> '{name}'")

    print(f"\nDone. Imported {imported} photo(s).")
    if skipped:
        print("Skipped:")
        for s in skipped:
            print(f"  - {s}")


if __name__ == "__main__":
    main()
