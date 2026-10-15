from __future__ import annotations

import json
import sys
import urllib.request


def post(base_url: str, path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def main() -> None:
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    add_payload = {
        "request_id": "smoke:chunk-0",
        "messages": [
            {
                "role": "user",
                "timestamp": 1704067200000,
                "content": "I adopted a cat named Luna.",
            }
        ],
        "user_id": "smoke:user",
        "session_id": "smoke:session",
    }
    added = post(base_url, "/v1/memories/add", add_payload)
    searched = post(
        base_url,
        "/v1/memories/search",
        {"query": "What is my cat called?", "user_id": "smoke:user", "top_k": 5},
    )
    assert added["success"] is True
    assert searched["data"] and "Luna" in searched["data"][0]["content"]
    print("ReFind Add/Search smoke test passed.")


if __name__ == "__main__":
    main()
