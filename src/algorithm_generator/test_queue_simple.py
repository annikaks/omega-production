import argparse
import os
import time

import requests
from dotenv import load_dotenv
from supabase import create_client


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple queue smoke test.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--count", type=int, default=5, help="Number of /generate calls")
    parser.add_argument("--sleep-s", type=float, default=1.0, help="Sleep between calls")
    return parser.parse_args()


def _get_supabase():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL or SUPABASE_KEY not set")
    return create_client(url, key)


def main() -> None:
    load_dotenv()
    args = _parse_args()
    supabase = _get_supabase()

    payload = {
        "description": "queue smoke test",
        "user_id": "queue_test_user",
        "creator_name": "queue_test",
    }

    for i in range(args.count):
        resp = requests.post(f"{args.base_url}/generate", json=payload, timeout=15)
        print(f"request {i + 1}/{args.count}: {resp.status_code} {resp.text}")
        time.sleep(args.sleep_s)

    res = (
        supabase.table("sandbox_queue")
        .select("id", count="exact")
        .eq("status", "queued")
        .execute()
    )
    queued = res.count or 0
    print(f"queued rows in supabase: {queued}")


if __name__ == "__main__":
    main()
