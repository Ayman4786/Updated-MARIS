import os


DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"


def get_backend_url() -> str:
    return os.getenv("MARIS_BACKEND_URL", DEFAULT_BACKEND_URL).rstrip("/")
