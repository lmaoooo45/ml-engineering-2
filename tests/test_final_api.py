from unittest import TestCase
from unittest.mock import Mock

from fastapi.testclient import TestClient
from movie_assistant.api import create_app
from scripts.evaluate_system import check_result


class FinalApiTests(TestCase):
    def test_dependency_session_and_deadline_status_codes(self):
        for error, expected in ((TimeoutError(),504),(RuntimeError("busy"),503),(ValueError("session"),400)):
            with self.subTest(expected=expected), TestClient(create_app(Mock(side_effect=error))) as client:
                self.assertEqual(client.post("/chat",json={"question":"test"}).status_code,expected)

    def test_evaluator_rejects_wrong_fact_seen_movie_and_wrong_text(self):
        movie={"id":1,"title":"Toy Story (1995)","year":1995,"genres":["Comedy"]}
        result={"status":"ok","text":"wrong","movies":[movie]}
        expected={"status":"ok","count":1,"genres":["Drama"],"genre_mode":"and","text_contains":["1995"]}
        errors=check_result(result,expected,{1:movie},{1})
        self.assertEqual(errors,["answer_text","genres","seen"])
