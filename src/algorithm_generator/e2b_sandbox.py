import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

load_dotenv()

class E2BSandboxError(RuntimeError):
    pass


def _load_sandbox_class():
    try:
        from e2b import Sandbox  # type: ignore
        return Sandbox
    except Exception:
        try:
            from e2b_code_interpreter import Sandbox  # type: ignore
            return Sandbox
        except Exception as exc:
            raise E2BSandboxError(
                "E2B SDK not available. Install e2b or e2b-code-interpreter."
            ) from exc


def _create_sandbox(Sandbox):
    api_key = (
        os.getenv("E2B_KEY")
        or os.getenv("E2B_API_KEY")
        or os.getenv("E2B_ACCESS_TOKEN")
    )
    template = os.getenv("E2B_TEMPLATE")
    if hasattr(Sandbox, "create"):
        try:
            return (
                Sandbox.create(template=template, api_key=api_key)
                if api_key
                else Sandbox.create(template=template)
            )
        except TypeError:
            return Sandbox.create()
    try:
        return Sandbox(api_key=api_key) if api_key else Sandbox()
    except TypeError:
        return Sandbox()


def _get_stream(execution: Any, name: str) -> str:
    direct = getattr(execution, name, None)
    if isinstance(direct, str):
        return direct
    logs = getattr(execution, "logs", None)
    if logs is not None:
        nested = getattr(logs, name, None)
        if isinstance(nested, str):
            return nested
    return ""


def _run_python_in_sandbox(sandbox: Any, code: str) -> Tuple[str, str]:
    if hasattr(sandbox, "run_code"):
        execution = sandbox.run_code(code)
        return _get_stream(execution, "stdout"), _get_stream(execution, "stderr")
    if hasattr(sandbox, "commands") and hasattr(sandbox.commands, "run"):
        wrapped = f"python - <<'PY'\n{code}\nPY"
        execution = sandbox.commands.run(wrapped)
        return _get_stream(execution, "stdout"), _get_stream(execution, "stderr")
    if hasattr(sandbox, "run"):
        wrapped = f"python - <<'PY'\n{code}\nPY"
        execution = sandbox.run(wrapped)
        return _get_stream(execution, "stdout"), _get_stream(execution, "stderr")
    raise E2BSandboxError("Unsupported E2B SDK interface.")


def _write_sandbox_file(sandbox: Any, path: str, content: str) -> None:
    if not hasattr(sandbox, "files"):
        raise E2BSandboxError("Sandbox filesystem API not available.")
    sandbox.files.write(path, content)


def _extract_result(stdout: str) -> Dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise E2BSandboxError("Failed to parse E2B sandbox output.")


def run_e2b_eval(
    code_string: str, class_name: str, dataset_names: List[str]
) -> Dict[str, float]:
    payload = {
        "code": code_string,
        "class_name": class_name,
        "dataset_names": dataset_names,
    }

    payload_json = json.dumps(payload)
    data_loader_path = Path(__file__).resolve().parent / "data_loader.py"
    if not data_loader_path.exists():
        raise E2BSandboxError("data_loader.py not found for sandbox upload.")
    data_loader_code = data_loader_path.read_text()
    runner = f"""
import json
import sys
import traceback
import types
import importlib.util

try:
    try:
        import numpy as np
        from sklearn.metrics import accuracy_score
        from sklearn.base import clone
    except Exception:
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "numpy", "scikit-learn", "openml", "pandas"]
        )
        import numpy as np
        from sklearn.metrics import accuracy_score
        from sklearn.base import clone

    payload = json.loads(open("/tmp/payload.json", "r").read())
    code_string = payload["code"]
    class_name = payload["class_name"]
    dataset_names = payload["dataset_names"]

    module = types.ModuleType("temp_mod")
    exec(code_string, module.__dict__)
    Cls = getattr(module, class_name)

    spec = importlib.util.spec_from_file_location("data_loader", "/tmp/data_loader.py")
    data_loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_loader)
    datasets = data_loader.load_classification_datasets(dataset_names)

    metrics = {{}}
    for ds in datasets:
        try:
            model = Cls()
            try:
                model = clone(model)
            except Exception:
                model = Cls()
            X_train, X_test, y_train, y_test = datasets[ds]
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            metrics[ds] = float(acc)
        except Exception:
            metrics[ds] = 0.0

    print(json.dumps({{"metrics": metrics}}))
except Exception as exc:
    print(json.dumps({{"metrics": {{}}, "error": f"{{type(exc).__name__}}: {{exc}}"}}))
"""

    Sandbox = _load_sandbox_class()
    sandbox = _create_sandbox(Sandbox)
    try:
        _write_sandbox_file(sandbox, "/tmp/payload.json", payload_json)
        _write_sandbox_file(sandbox, "/tmp/data_loader.py", data_loader_code)
        stdout, stderr = _run_python_in_sandbox(sandbox, runner)
    finally:
        try:
            sandbox.close()
        except Exception:
            pass

    if stderr:
        # Keep stderr available for troubleshooting, but don't fail on warnings.
        pass

    result = _extract_result(stdout)
    if "error" in result:
        raise E2BSandboxError(result["error"])
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        raise E2BSandboxError("E2B sandbox did not return metrics.")
    return {k: float(v) for k, v in metrics.items()}


def test_e2b_sandbox() -> Dict[str, str]:
    Sandbox = _load_sandbox_class()
    sandbox = _create_sandbox(Sandbox)
    try:
        stdout, stderr = _run_python_in_sandbox(
            sandbox, "print('e2b_ok'); import sys; print(sys.version.split()[0])"
        )
    finally:
        try:
            sandbox.close()
        except Exception:
            pass
    return {"stdout": stdout.strip(), "stderr": stderr.strip()}


if __name__ == "__main__":
    result = test_e2b_sandbox()
    print(json.dumps(result))
