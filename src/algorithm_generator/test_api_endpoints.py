import os
import unittest
from types import SimpleNamespace
from unittest import mock

from fastapi.testclient import TestClient

os.environ.setdefault("SUPABASE_URL", "http://example.com")
os.environ.setdefault("SUPABASE_KEY", "test")
os.environ.setdefault("SUPABASE_ANON_KEY", "anon")
os.environ.setdefault("ANTHROPIC_API_KEY", "test")

import api as api_module


class ApiEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        api_module.app.router.on_startup.clear()
        api_module.app.router.on_shutdown.clear()

    def setUp(self) -> None:
        self.client = TestClient(api_module.app)

    def test_my_info_requires_auth_header(self) -> None:
        res = self.client.get("/my_info")
        self.assertEqual(res.status_code, 401)

    def test_my_info_returns_user_id_and_creator_name(self) -> None:
        fake_user = SimpleNamespace(
            id="user_123",
            email="user@example.com",
            user_metadata={"display_name": "Ada"},
        )
        with mock.patch.object(
            api_module.supabase.auth, "get_user", return_value=SimpleNamespace(user=fake_user)
        ):
            res = self.client.get("/my_info", headers={"Authorization": "Bearer token"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"user_id": "user_123", "creator_name": "Ada"})

    def test_algorithm_code_by_class_not_found(self) -> None:
        mock_table = mock.Mock()
        mock_table.select.return_value = mock_table
        mock_table.eq.return_value = mock_table
        mock_table.single.return_value = mock_table
        mock_table.execute.return_value = SimpleNamespace(data=None)
        with mock.patch.object(api_module.supabase, "table", return_value=mock_table):
            res = self.client.get("/algorithm-code/by-class/MissingModel")
        self.assertEqual(res.status_code, 404)

    def test_algorithm_code_by_class_returns_code(self) -> None:
        mock_table = mock.Mock()
        mock_table.select.return_value = mock_table
        mock_table.eq.return_value = mock_table
        mock_table.single.return_value = mock_table
        mock_table.execute.return_value = SimpleNamespace(
            data={"class_name": "MyModel", "file_name": "my_model.py", "algorithm_code": "print('hi')\n"}
        )
        with mock.patch.object(api_module.supabase, "table", return_value=mock_table):
            res = self.client.get("/algorithm-code/by-class/MyModel")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/plain", res.headers.get("content-type", ""))
        self.assertEqual(res.text, "print('hi')\n")

    def test_generate_queues_job(self) -> None:
        os.environ["USE_E2B_SANDBOX"] = "1"
        fake_queue = mock.Mock()
        fake_queue.enqueue_job.return_value = {"status": "queued", "job_id": "job_1", "position": 3}
        api_module.queue_manager = fake_queue
        payload = {"description": "Test model", "user_id": "user_123", "creator_name": "Ada"}
        res = self.client.post("/generate", json=payload)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json().get("status"), "queued")


if __name__ == "__main__":
    unittest.main()
