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
    timeout_env = os.getenv("E2B_SANDBOX_TIMEOUT")
    timeout = None
    if timeout_env:
        try:
            timeout = int(timeout_env)
        except ValueError:
            timeout = None
    if hasattr(Sandbox, "create"):
        try:
            return (
                Sandbox.create(template=template, api_key=api_key, timeout=timeout)
                if api_key
                else Sandbox.create(template=template, timeout=timeout)
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
    timeout_env = os.getenv("E2B_CODE_TIMEOUT")
    timeout = None
    if timeout_env is not None:
        try:
            timeout = int(timeout_env)
        except ValueError:
            timeout = None
    if hasattr(sandbox, "run_code"):
        try:
            execution = sandbox.run_code(code, timeout=timeout)
        except TypeError:
            execution = sandbox.run_code(code)
        return _get_stream(execution, "stdout"), _get_stream(execution, "stderr")
    if hasattr(sandbox, "commands") and hasattr(sandbox.commands, "run"):
        wrapped = f"python - <<'PY'\n{code}\nPY"
        try:
            execution = sandbox.commands.run(wrapped, timeout=timeout)
        except TypeError:
            execution = sandbox.commands.run(wrapped)
        return _get_stream(execution, "stdout"), _get_stream(execution, "stderr")
    if hasattr(sandbox, "run"):
        wrapped = f"python - <<'PY'\n{code}\nPY"
        try:
            execution = sandbox.run(wrapped, timeout=timeout)
        except TypeError:
            execution = sandbox.run(wrapped)
        return _get_stream(execution, "stdout"), _get_stream(execution, "stderr")
    raise E2BSandboxError("Unsupported E2B SDK interface.")


def _write_sandbox_file(sandbox: Any, path: str, content: str) -> None:
    if not hasattr(sandbox, "files"):
        raise E2BSandboxError("Sandbox filesystem API not available.")
    sandbox.files.write(path, content)


def create_e2b_sandbox() -> Any:
    Sandbox = _load_sandbox_class()
    return _create_sandbox(Sandbox)


def close_e2b_sandbox(sandbox: Any) -> None:
    try:
        sandbox.close()
    except Exception:
        pass


def _extract_result(stdout: str) -> Dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise E2BSandboxError("Failed to parse E2B sandbox output.")


def _build_eval_runner() -> str:
    return """
import json
import sys
import traceback
import types
import importlib.util
import os

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

    def _safe_name(name):
        return name.replace(" ", "_").replace("/", "_")

    def _load_cached(dataset_names):
        cached = {}
        missing = []
        for name in dataset_names:
            path = f"/data/benchmarks/{_safe_name(name)}.npz"
            if os.path.exists(path):
                data = np.load(path)
                cached[name] = (
                    data["X_train"],
                    data["X_test"],
                    data["y_train"],
                    data["y_test"],
                )
            else:
                missing.append(name)
        return cached, missing

    payload = json.loads(open("/tmp/payload.json", "r").read())
    code_string = payload["code"]
    class_name = payload["class_name"]
    dataset_names = payload["dataset_names"]

    module = types.ModuleType("temp_mod")
    exec(code_string, module.__dict__)
    Cls = getattr(module, class_name)

    cached, missing = _load_cached(dataset_names)
    datasets = cached
    if missing:
        spec = importlib.util.spec_from_file_location("data_loader", "/tmp/data_loader.py")
        data_loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(data_loader)
        datasets.update(data_loader.load_classification_datasets(missing))

    metrics = {}
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

    print(json.dumps({"metrics": metrics}))
except Exception as exc:
    print(json.dumps({"metrics": {}, "error": f"{type(exc).__name__}: {exc}"}))
"""


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
    runner = _build_eval_runner()

    sandbox = create_e2b_sandbox()
    try:
        stdout, stderr = run_e2b_eval_in_sandbox(
            sandbox,
            payload_json,
            data_loader_code,
            runner,
        )
    finally:
        close_e2b_sandbox(sandbox)

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


def run_e2b_eval_in_sandbox(
    sandbox: Any,
    payload_json: str,
    data_loader_code: str,
    runner: str,
) -> Tuple[str, str]:
    _write_sandbox_file(sandbox, "/tmp/payload.json", payload_json)
    _write_sandbox_file(sandbox, "/tmp/data_loader.py", data_loader_code)
    return _run_python_in_sandbox(sandbox, runner)


def eval_with_sandbox(
    sandbox: Any,
    code_string: str,
    class_name: str,
    dataset_names: List[str],
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
    runner = _build_eval_runner()

    stdout, _stderr = run_e2b_eval_in_sandbox(
        sandbox,
        payload_json,
        data_loader_code,
        runner,
    )
    result = _extract_result(stdout)
    if "error" in result:
        raise E2BSandboxError(result["error"])
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        raise E2BSandboxError("E2B sandbox did not return metrics.")
    return {k: float(v) for k, v in metrics.items()}


def _build_generate_runner() -> str:
    return """
import json
import re
import sys
import traceback
import types
import importlib.util
import os

try:
    try:
        import numpy as np
        from sklearn.metrics import accuracy_score
        from sklearn.base import clone
    except Exception:
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "numpy", "scikit-learn"]
        )
        import numpy as np
        from sklearn.metrics import accuracy_score
        from sklearn.base import clone

    try:
        import anthropic
    except Exception:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "anthropic"])
        import anthropic

    def _safe_name(name):
        return name.replace(" ", "_").replace("/", "_")

    def _load_cached(dataset_names):
        cached = {}
        missing = []
        for name in dataset_names:
            path = f"/data/benchmarks/{_safe_name(name)}.npz"
            if os.path.exists(path):
                data = np.load(path)
                cached[name] = (
                    data["X_train"],
                    data["X_test"],
                    data["y_train"],
                    data["y_test"],
                )
            else:
                missing.append(name)
        return cached, missing

    payload = json.loads(open("/tmp/payload.json", "r").read())
    description = payload["description"]
    dataset_names = payload["dataset_names"]
    api_key = payload["anthropic_api_key"]

    client = anthropic.Anthropic(api_key=api_key)
    mega_prompt = f\"\"\"
    Design a {description} classifier in the style of SciKit learn.

    1. Provide a succinct pythonic class name between <class_name></class_name> tags.
    2. Provide a succinct pythonic filename (ending in .py) between <file_name></file_name> tags.
    3. Provide the complete implementation in a single markdown python code block.

    The class must inherit from sklearn.base.BaseEstimator.
    Example:
    from sklearn.base import BaseEstimator
    class <class_name>(BaseEstimator):
        def __init__(self, ...):
            # All arguments must be saved as attributes

    The class must implement fit(self, X_train, y_train) and predict(self, X_test).
    Only return the tags and the code block.
    \"\"\"
    message = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=4000,
        temperature=0,
        system="You are a world-class research engineer.",
        messages=[{"role": "user", "content": [{"type": "text", "text": mega_prompt}]}],
    )
    response = message.content[0].text

    class_name = re.search(r"<class_name>(.*?)</class_name>", response, re.DOTALL).group(1).strip()
    file_name = re.search(r"<file_name>(.*?)</file_name>", response, re.DOTALL).group(1).strip()
    snippets = re.findall(r"```(?:python)?\\n(.*?)```", response, re.DOTALL)
    code_string = snippets[0].strip() if snippets else ""

    cached, missing = _load_cached(dataset_names)
    datasets = cached
    if missing:
        spec = importlib.util.spec_from_file_location("data_loader", "/tmp/data_loader.py")
        data_loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(data_loader)
        datasets.update(data_loader.load_classification_datasets(missing))

    module = types.ModuleType("temp_mod")
    exec(code_string, module.__dict__)
    Cls = getattr(module, class_name)

    metrics = {}
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

    print(
        json.dumps(
            {
                "metrics": metrics,
                "class_name": class_name,
                "file_name": file_name,
                "code_string": code_string,
                "strategy": "E2B Synthesis",
            }
        )
    )
except Exception as exc:
    print(json.dumps({"metrics": {}, "error": f"{type(exc).__name__}: {exc}"}))
"""


def run_e2b_generate_and_eval(
    description: str,
    dataset_names: List[str],
    anthropic_api_key: str,
) -> Dict[str, Any]:
    payload = {
        "description": description,
        "dataset_names": dataset_names,
        "anthropic_api_key": anthropic_api_key,
    }

    payload_json = json.dumps(payload)
    data_loader_path = Path(__file__).resolve().parent / "data_loader.py"
    if not data_loader_path.exists():
        raise E2BSandboxError("data_loader.py not found for sandbox upload.")
    data_loader_code = data_loader_path.read_text()
    runner = _build_generate_runner()

    sandbox = create_e2b_sandbox()
    try:
        stdout, stderr = run_e2b_generate_and_eval_in_sandbox(
            sandbox,
            payload_json,
            data_loader_code,
            runner,
        )
    finally:
        close_e2b_sandbox(sandbox)

    result = _extract_result(stdout)
    if "error" in result:
        raise E2BSandboxError(result["error"])
    return result


def run_e2b_generate_and_eval_in_sandbox(
    sandbox: Any,
    payload_json: str,
    data_loader_code: str,
    runner: str,
) -> Tuple[str, str]:
    _write_sandbox_file(sandbox, "/tmp/payload.json", payload_json)
    _write_sandbox_file(sandbox, "/tmp/data_loader.py", data_loader_code)
    return _run_python_in_sandbox(sandbox, runner)


def generate_and_eval_with_sandbox(
    sandbox: Any,
    description: str,
    dataset_names: List[str],
    anthropic_api_key: str,
) -> Dict[str, Any]:
    payload = {
        "description": description,
        "dataset_names": dataset_names,
        "anthropic_api_key": anthropic_api_key,
    }
    payload_json = json.dumps(payload)
    data_loader_path = Path(__file__).resolve().parent / "data_loader.py"
    if not data_loader_path.exists():
        raise E2BSandboxError("data_loader.py not found for sandbox upload.")
    data_loader_code = data_loader_path.read_text()
    runner = _build_generate_runner()

    stdout, _stderr = run_e2b_generate_and_eval_in_sandbox(
        sandbox,
        payload_json,
        data_loader_code,
        runner,
    )
    result = _extract_result(stdout)
    if "error" in result:
        raise E2BSandboxError(result["error"])
    return result


def test_e2b_sandbox() -> Dict[str, str]:
    Sandbox = _load_sandbox_class()
    sandbox = _create_sandbox(Sandbox)
    try:
        stdout, stderr = _run_python_in_sandbox(
            sandbox,
            "import json\n"
            "import sys\n"
            "payload = {\n"
            "  'python': sys.version.split()[0],\n"
            "}\n"
            "try:\n"
            "  import sklearn\n"
            "  import openml\n"
            "  payload['sklearn'] = sklearn.__version__\n"
            "  payload['openml'] = openml.__version__\n"
            "  payload['openml_cache_dir'] = getattr(openml.config, 'cache_directory', None)\n"
            "except Exception as exc:\n"
            "  payload['error'] = f\"{type(exc).__name__}: {exc}\"\n"
            "print(json.dumps(payload))\n",
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
