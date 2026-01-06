import os
import unittest
from datetime import datetime
from unittest import mock

from sandbox_queue import SandboxQueueManager, E2BSandboxError


class _NoopSupabase:
    def table(self, _name):
        raise AssertionError("Supabase should not be called in this test.")


class _CaptureManager(SandboxQueueManager):
    def __init__(self):
        super().__init__(_NoopSupabase(), pool_size=1, queue_limit=1, job_timeout_s=10, worker_count=1)
        self.failed = None

    def _mark_failed(self, job_id: str, error: str) -> None:
        self.failed = (job_id, error)


class _DummyResponse:
    def __init__(self, data):
        self.data = data


class _DummyTable:
    def __init__(self, row):
        self.row = row

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def single(self):
        return self

    def execute(self):
        return _DummyResponse(self.row)


class _DummySupabase:
    def __init__(self, row):
        self.row = row

    def table(self, _name):
        return _DummyTable(self.row)


class SandboxQueueErrorHandlingTests(unittest.TestCase):
    def test_process_job_marks_failed_on_e2b_error(self):
        manager = _CaptureManager()
        job = {
            "id": "job-123",
            "description": "test",
            "user_id": None,
            "creator_name": "tester",
            "created_at": datetime.utcnow().isoformat(),
        }
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        with mock.patch("sandbox_queue.generate_and_eval_with_sandbox", side_effect=E2BSandboxError("boom")):
            manager._process_job(job, sandbox=None)
        self.assertIsNotNone(manager.failed)
        self.assertEqual(manager.failed[0], "job-123")
        self.assertIn("boom", manager.failed[1])

    def test_get_job_status_failed_includes_error(self):
        row = {
            "id": "job-err",
            "status": "failed",
            "error": "sandbox failed",
            "algorithm_id": None,
            "algorithm_id_text": None,
        }
        manager = SandboxQueueManager(_DummySupabase(row), pool_size=1, queue_limit=1, job_timeout_s=10, worker_count=1)
        status = manager.get_job_status("job-err")
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["error"], "sandbox failed")


if __name__ == "__main__":
    unittest.main()
