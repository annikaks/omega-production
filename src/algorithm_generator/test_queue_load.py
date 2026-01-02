import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List

import requests


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load test queue with concurrent /generate calls.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--count", type=int, default=20, help="Number of concurrent requests")
    parser.add_argument("--user-id", default="queue_test_user", help="User id to send")
    parser.add_argument("--creator-name", default="queue_test", help="Creator name to send")
    parser.add_argument(
        "--description",
        default="baseline classifier",
        help="Prompt/description for synthesis",
    )
    return parser.parse_args()


def _post_generate(base_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(f"{base_url}/generate", json=payload, timeout=30)
    try:
        data = resp.json()
    except Exception:
        data = {"raw": resp.text}
    return {"status_code": resp.status_code, "data": data}


def run_load_test(base_url: str, count: int, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = []
    with ThreadPoolExecutor(max_workers=count) as executor:
        futures = [executor.submit(_post_generate, base_url, payload) for _ in range(count)]
        for fut in as_completed(futures):
            results.append(fut.result())
    return results


if __name__ == "__main__":
    args = _parse_args()
    payload = {
        "description": args.description,
        "user_id": args.user_id,
        "creator_name": args.creator_name,
    }
    results = run_load_test(args.base_url, args.count, payload)
    print(json.dumps(results, indent=2))
