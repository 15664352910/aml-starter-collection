"""End-to-end smoke against a running service, with the platform's own strictness.

The platform fails a stage immediately on a malformed response *even under HTTP
200*, so this script asserts the same invariants locally before we spend the
once-per-three-months ``full`` quota.

    # terminal 1
    EMBED_ENABLED=false uvicorn src.app:app --port 8000
    # terminal 2
    python scripts/smoke_local.py --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

USER = "eval:smoke:locomo:conv-0"
SESSION = "eval:smoke:sample:0"

CONVERSATION = [
    ("user", 1704067200000, "Hi Rob! I just got back from Stockholm, Sweden last week."),
    ("assistant", 1704067260000, "Welcome back! How was the trip to Sweden?"),
    ("user", 1704067320000, "Wonderful. I visited the Vasa Museum and ate way too much herring."),
    ("assistant", 1704067380000, "The Vasa is remarkable. Did you see the Nobel Museum too?"),
    ("user", 1704153600000, "No, I skipped that one. My sister Freya is training for the Berlin marathon."),
    ("user", 1704240000000, "Actually I changed my mind — I do want to visit the Nobel Museum next time."),
    ("user", 1704326400000, "We adopted a tabby cat named Miso last spring."),
    ("assistant", 1704326460000, "Miso is a great name for a cat."),
]

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok    {message}")
    else:
        print(f"  FAIL  {message}")
        FAILURES.append(message)


def post(
    base_url: str, path: str, payload: dict, api_key: str | None, scheme: str = "bearer"
) -> tuple[int, dict]:
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        if scheme == "bearer":
            headers["Authorization"] = f"Bearer {api_key}"
        elif scheme == "token":
            headers["Authorization"] = f"Token {api_key}"
        elif scheme == "x-api-key":
            headers["X-Api-Key"] = api_key
        elif scheme == "raw":
            headers["Authorization"] = api_key
        else:
            raise SystemExit(f"unknown --auth-scheme {scheme!r}")
    request = urllib.request.Request(base_url + path, data=body, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        raw = error.read()
        try:
            return error.code, json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return error.code, {"raw": raw.decode(errors="replace")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--auth-scheme",
        default="bearer",
        choices=["bearer", "token", "x-api-key", "raw"],
        help="How to carry --api-key. AUTH_MODE=auto accepts all of these.",
    )
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    def send(path: str, payload: dict) -> tuple[int, dict]:
        return post(base, path, payload, args.api_key, args.auth_scheme)

    print("health")
    try:
        with urllib.request.urlopen(base + "/health", timeout=30) as response:
            check(200 <= response.status < 300, f"GET /health returned {response.status}")
    except Exception as error:  # noqa: BLE001
        check(False, f"GET /health failed: {error}")
        return 1

    # Chunk exactly as the platform does: one Add per source session, split at 20
    # messages / 2000 words.
    print("\nadd (synchronous write)")
    chunks = [CONVERSATION[i : i + 20] for i in range(0, len(CONVERSATION), 20)]
    for position, chunk in enumerate(chunks):
        request_id = f"eval:smoke:locomo_refined:conv-0:chunk-{position}"
        payload = {
            "request_id": request_id,
            "messages": [
                {"role": role, "timestamp": stamp, "content": text} for role, stamp, text in chunk
            ],
            "user_id": USER,
            "session_id": SESSION,
        }
        started = time.time()
        status, body = send("/add", payload)
        elapsed = time.time() - started

        check(status == 200, f"chunk-{position} HTTP 200 (got {status}, {elapsed:.2f}s)")
        check(body.get("success") is True, f"chunk-{position} success is boolean true")
        check(body.get("request_id") == request_id, f"chunk-{position} request_id echoed exactly")
        check(body.get("user_id") == USER, f"chunk-{position} user_id echoed exactly")
        check(body.get("session_id") == SESSION, f"chunk-{position} session_id echoed exactly")
        check("task_id" not in body, f"chunk-{position} no task_id in response")

    print("\nadd idempotency (Add is retried on 408/409/425/429/5xx)")
    replay = {
        "request_id": "eval:smoke:locomo_refined:conv-0:chunk-0",
        "messages": [
            {"role": role, "timestamp": stamp, "content": text}
            for role, stamp, text in CONVERSATION[:20]
        ],
        "user_id": USER,
        "session_id": SESSION,
    }
    status, body = send("/add", replay)
    check(status == 200 and body.get("success") is True, "replayed Add still returns success")

    print("\nsearch (open-ended)")
    cases = [
        ("Which country did the user visit recently?", "Sweden"),
        ("Who is training for a marathon?", "Freya"),
        ("What is the name of the cat?", "Miso"),
    ]
    for query, expected in cases:
        status, body = send("/search", {"query": query, "user_id": USER, "top_k": args.top_k})
        check(status == 200, f"HTTP 200 for {query!r}")
        check(isinstance(body, dict) and "data" in body, "response is an object with data")
        data = body.get("data") if isinstance(body, dict) else None
        check(isinstance(data, list), "data is an array")
        if not isinstance(data, list):
            continue
        check(len(data) <= args.top_k, f"returned {len(data)} <= top_k={args.top_k}")
        check(
            all(isinstance(i.get("id"), str) and i.get("id") for i in data),
            "every item has a non-empty id",
        )
        check(
            all(isinstance(i.get("content"), str) and i.get("content") for i in data),
            "every item has non-empty content",
        )
        blob = " ".join(i.get("content", "") for i in data)
        check(expected in blob, f"{expected!r} retrieved for {query!r}")

    print("\nsearch (multiple choice, options present)")
    status, body = send(
        "/search",
        {
            "query": "Which museum does the user now want to visit?",
            "options": ["A. Nobel Museum", "B. Vasa Museum", "C. British Museum"],
            "user_id": USER,
            "top_k": args.top_k,
        },
    )
    check(status == 200, "HTTP 200 with options")
    blob = " ".join(i.get("content", "") for i in body.get("data", []))
    check("Nobel" in blob, "option-augmented retrieval surfaced the Nobel Museum turn")

    print("\ntimestamps in content (the answer model resolves relative dates from these)")
    status, body = send("/search", {"query": "Stockholm Sweden", "user_id": USER, "top_k": 5})
    top = (body.get("data") or [{}])[0]
    check("2024-01-01" in top.get("content", ""), f"timestamp present: {top.get('content','')[:80]!r}")

    print("\nuser_id isolation")
    status, body = send("/search", {"query": "Stockholm Freya Miso", "user_id": "eval:smoke:other:conv-9", "top_k": 100})
    check(body.get("data") == [], "unknown user_id returns an empty array, not another user's memories")

    print()
    if FAILURES:
        print(f"SMOKE FAILED — {len(FAILURES)} check(s):")
        for item in FAILURES:
            print(f"  - {item}")
        return 1
    print("SMOKE PASSED — every contract invariant held.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
