from frontend.api.client import MarisClient


class Response:
    def __init__(self, payload, ok=True):
        self._payload = payload
        self.ok = ok

    def json(self):
        return self._payload


def test_client_accepts_list_payloads_for_library_and_history():
    assert MarisClient._parse_response(Response([]), "failed") == []
