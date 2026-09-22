# Face Verification Webapp

A Flask app that verifies a face captured from the browser's webcam against a
database of enrolled people, using [InsightFace](https://github.com/deepinsight/insightface)
(ArcFace embeddings) for recognition.

This started as a small single-file demo (`app.py` + a flat `faces_db/` folder
of photos). It's now a small full-stack app:

- **SQLite-backed storage** — people, multiple photos/embeddings per person,
  a full verification history log, and tunable settings, instead of files on disk.
- **Browser-based enrollment** — register new people with a live webcam capture
  flow (multiple angles) instead of manually dropping JPEGs into a folder.
- **Liveness heuristics** — a blur check, a face-size sanity check, and an
  optional two-frame blink challenge, to make it a bit harder to spoof with a
  printed photo. **This is not a certified anti-spoofing system** — see the
  caveat below.
- **Admin panel** — a login-protected dashboard to manage enrolled people,
  browse verification history, and tune the match threshold.
- **JSON API** — API-key protected endpoints for programmatic verification.
- **AJAX frontend** — capture and verify without a full page reload, live
  confidence bar, and an annotated result image.

## Quick start

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env            # optional: edit thresholds, etc.
python app.py
```

Open `http://localhost:5000`. On first run the app prints a generated admin
username/password and API key to the console — save these, or set
`ADMIN_PASSWORD` / `API_KEY` in `.env` to choose your own.

### Migrating your old `faces_db/` photos

If you have photos in the old `faces_db/person1.jpg` style (name inferred from
the filename with trailing digits stripped), run:

```bash
python seed_from_faces_db.py
```

This imports each photo into the new database as a person + embedding.

## Project layout

```
app.py                 App factory: wires config, DB, face engine, blueprints
config.py               All settings, loaded from environment variables
database.py              SQLite schema + a small DB helper class
face_engine.py            insightface wrapper: detection, matching, liveness heuristics
main_routes.py             Public "/" verify page + /verify AJAX endpoint
enroll_routes.py            "/enroll" page + submit endpoint
admin_routes.py              Admin dashboard, person management, history, settings
auth.py                       Session-based admin login/logout + require_admin decorator
api_routes.py                  JSON API under /api/v1, protected by X-API-Key
seed_from_faces_db.py           One-time migration script for old faces_db/*.jpg
test_core.py                    Unit tests for the DB layer + matching math (no camera/model needed)
templates/                      Jinja templates (base, verify, enroll, admin pages)
static/                          CSS + shared camera.js helper
```

## How matching works

Every enrolled photo is stored as a 512-d ArcFace embedding. On verification,
the captured face's embedding is compared via cosine similarity against every
stored embedding; the best match's score is compared to two settings
(editable in Admin → Settings):

- **`match_threshold`** (default 0.40): scores at or above this are a confident match.
- **`review_margin`** (default 0.07): scores within this margin below the
  threshold are flagged `review` (possible match, worth a human look) instead
  of outright `unknown`.

## Liveness checks — read this

The "liveness" feature here is a **heuristic pipeline**, not a certified
anti-spoofing system:

1. **Blur check** — rejects captures that are unusually blurry (variance of
   the Laplacian below a threshold), which catches some low-quality photo
   spoofs.
2. **Face-size check** — rejects faces that are implausibly small or large
   relative to the frame.
3. **Blink challenge** (optional, two frames) — compares eye-openness between
   two captures taken ~650ms apart, using the model's 106-point landmarks
   (falls back to skipping this check if the loaded model doesn't expose
   dense landmarks).

None of this defeats a determined attacker with a video replay or a 3D mask.
If you need real anti-spoofing guarantees, pair this with a dedicated
liveness SDK or depth/IR camera hardware.

## API usage

```bash
curl -X POST http://localhost:5000/api/v1/verify \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your key from Admin > Settings>" \
  -d '{"image": "data:image/jpeg;base64,<...>"}'
```

Returns `{"decision": "matched"|"review"|"unknown"|"no_face"|"liveness_fail", "name": ..., "score": ...}`.

## Tests

`test_core.py` covers the SQLite layer and the pure-numpy matching/liveness
math without needing the `insightface` model or a real camera. Run with:

```bash
pip install pytest
pytest test_core.py -v
```

## Security notes for real deployments

- Change `SECRET_KEY`, `ADMIN_PASSWORD`, and `API_KEY` from their generated
  defaults before exposing this beyond localhost.
- Run behind HTTPS — face images and session cookies should not travel in
  plaintext.
- The dev server (`app.run(...)`) is for local use only; use a production
  WSGI server (gunicorn, waitress, etc.) for anything real.
- Captured images are stored on disk under `uploads/` for auditing/history —
  add a retention/deletion policy if that matters for your use case.
