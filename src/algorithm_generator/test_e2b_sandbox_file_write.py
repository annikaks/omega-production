import unittest

from e2b_sandbox import _write_sandbox_file, E2BSandboxError


class _FailingFiles:
    def __init__(self, fail_times):
        self.fail_times = fail_times
        self.calls = 0

    def write(self, _path, _content):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("write failed")


class _Sandbox:
    def __init__(self, fail_times):
        self.files = _FailingFiles(fail_times)


class SandboxFileWriteTests(unittest.TestCase):
    def test_write_retries_then_succeeds(self):
        sandbox = _Sandbox(fail_times=2)
        _write_sandbox_file(sandbox, "/tmp/payload.json", "{}")
        self.assertEqual(sandbox.files.calls, 3)

    def test_write_raises_friendly_error(self):
        sandbox = _Sandbox(fail_times=3)
        with self.assertRaises(E2BSandboxError) as ctx:
            _write_sandbox_file(sandbox, "/tmp/payload.json", "{}")
        self.assertIn("Sandbox file upload failed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
