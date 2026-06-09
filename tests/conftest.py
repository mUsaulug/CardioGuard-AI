"""Shared pytest fixtures."""

import os

import pytest
from fastapi.testclient import TestClient

# Reduce native-library thread contention during model load in tests.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from src.backend.main import app


@pytest.fixture(scope="module")
def client():
    """TestClient with lifespan enabled so models load on startup."""
    with TestClient(app) as test_client:
        yield test_client
