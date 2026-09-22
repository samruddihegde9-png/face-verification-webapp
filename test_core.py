"""
Unit tests covering everything that doesn't require the insightface model or a
real camera: the SQLite layer and the pure-numpy matching math.

Run with:  pytest test_core.py
"""
import os
import tempfile
import numpy as np
import pytest

from database import DB, bootstrap_defaults
from config import Config
from face_engine import cosine_similarity, FaceEngine


class FakeFace:
    def __init__(self, bbox):
        self.bbox = np.array(bbox, dtype=np.float32)


def test_cosine_similarity_identical_vectors():
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector_is_safe():
    a = np.zeros(3, dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert cosine_similarity(a, b) == 0.0


def test_best_match_picks_closest_vector():
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    names = ["alice", "bob"]
    ids = [1, 2]
    vectors = [np.array([0.0, 1.0, 0.0], dtype=np.float32), np.array([0.99, 0.01, 0.0], dtype=np.float32)]
    name, pid, score = FaceEngine.best_match(query, names, ids, vectors)
    assert name == "bob"
    assert pid == 2
    assert score > 0.9


def test_best_match_empty_db():
    name, pid, score = FaceEngine.best_match(np.array([1.0, 0.0]), [], [], [])
    assert name is None and pid is None and score == 0.0


def test_check_face_size_ratio():
    face = FakeFace([10, 10, 60, 60])  # 50x50 = 2500 px face
    ok, ratio = FaceEngine.check_face_size(face, (100, 100, 3), min_ratio=0.1, max_ratio=0.9)
    assert ratio == pytest.approx(0.25)
    assert ok is True

    ok2, ratio2 = FaceEngine.check_face_size(face, (100, 100, 3), min_ratio=0.5, max_ratio=0.9)
    assert ok2 is False


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config()
        cfg.DB_PATH = os.path.join(tmp, "test.db")
        db = DB(cfg.DB_PATH)
        bootstrap_defaults(db, cfg)
        yield db


def test_create_person_and_embedding(temp_db):
    pid = temp_db.create_person("Alice")
    temp_db.add_embedding(pid, np.random.rand(512).astype(np.float32), image_path="x.jpg")
    names, ids, vecs = temp_db.all_embeddings()
    assert names == ["Alice"]
    assert ids == [pid]
    assert vecs[0].shape == (512,)


def test_delete_person_cascades_embeddings(temp_db):
    pid = temp_db.create_person("Bob")
    temp_db.add_embedding(pid, np.random.rand(512).astype(np.float32))
    temp_db.delete_person(pid)
    names, ids, vecs = temp_db.all_embeddings()
    assert names == []


def test_settings_roundtrip(temp_db):
    temp_db.set_setting("match_threshold", 0.55)
    assert temp_db.get_setting("match_threshold") == "0.55"


def test_bootstrap_generates_admin_and_api_key(temp_db):
    assert temp_db.get_setting("admin_password_hash") is not None
    assert temp_db.get_setting("api_key") is not None


def test_verification_log_roundtrip(temp_db):
    temp_db.add_log(person_id=None, matched_name=None, score=0.1, decision="unknown",
                     liveness_passed=True, liveness_detail="ok", image_path=None,
                     source="web", ip_address="127.0.0.1")
    logs = temp_db.recent_logs(limit=10)
    assert len(logs) == 1
    assert logs[0]["decision"] == "unknown"
