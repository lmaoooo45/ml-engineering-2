import unittest
from unittest.mock import Mock

from fastapi.testclient import TestClient

from movie_assistant.api import create_app
from movie_assistant.gateway import allowed_request
from movie_assistant.ollama_client import MODELS


class GatewayTests(unittest.TestCase):
    def test_forbidden_model_operations_rejected(self):
        self.assertFalse(allowed_request("/api/pull", {"model": MODELS["gemma"]}))
        self.assertFalse(allowed_request("/api/chat", {"model": ["bad"]}))
        self.assertFalse(allowed_request("https://example.com", None))
        self.assertFalse(allowed_request("/api/chat", {
            "model": MODELS["gemma"], "stream": False, "keep_alive": 0,
            "options": {"num_ctx": 999999, "num_predict": 512},
        }))

    def test_api_rejects_input_before_starting_worker(self):
        transport = Mock()
        with TestClient(create_app(transport)) as client:
            response = client.post("/chat", json={"question": "   "})
            self.assertEqual(response.status_code, 422)
            transport.assert_not_called()

    def test_api_passes_validated_request_to_same_transport(self):
        transport = Mock(return_value={"status": "ok", "movies": []})
        with TestClient(create_app(transport)) as client:
            self.assertEqual(client.get("/docs").status_code, 200)
            response = client.post("/chat", json={"question": "Какого года Toy Story?"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(transport.call_args.args[0], "chat")
            self.assertEqual(transport.call_args.args[1]["question"], "Какого года Toy Story?")
