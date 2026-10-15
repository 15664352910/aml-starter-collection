"""端到端 HTTP 冒烟检查：验证 AML Add/Search 契约（/aml/add、/aml/search）。

用法：
    python scripts/smoke_check.py [base_url] [key]

本地无认证跑：  python scripts/smoke_check.py http://127.0.0.1:8000
线上带 key 跑： python scripts/smoke_check.py https://<your-host> <memory-system-key>
"""

from __future__ import annotations

import sys
import time

import httpx

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global failures
    print(f"[{PASS if ok else FAIL}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures += 1


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    key = sys.argv[2] if len(sys.argv) > 2 else ""
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    ts = int(time.time() * 1000)

    with httpx.Client(base_url=base, headers=headers, timeout=30.0) as client:
        r = client.get("/health")
        check("GET /health 200", r.status_code == 200, r.text[:120])

        payload = {
            "request_id": "smoke:conv-0:chunk-0",
            "messages": [
                {"role": "user", "content": "I moved to Porto two weeks ago.", "timestamp": ts - 14 * 86400_000},
                {"role": "assistant", "content": "Porto is a lovely city!", "timestamp": ts - 14 * 86400_000 + 30_000},
                {"role": "user", "content": "I love hiking, the coastal trails are beautiful.", "timestamp": ts - 3 * 86400_000},
                {"role": "user", "content": "My visa appointment is on November 3rd.", "timestamp": ts - 86400_000},
            ],
            "user_id": "smoke:user-a",
            "session_id": "smoke:session:0",
        }
        r = client.post("/aml/add", json=payload)
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        check(
            "POST /aml/add 200 + 契约字段回显",
            r.status_code == 200
            and body.get("success") is True
            and body.get("request_id") == payload["request_id"]
            and body.get("user_id") == payload["user_id"]
            and body.get("session_id") == payload["session_id"],
            f"http={r.status_code} body={r.text[:140]}",
        )

        r2 = client.post(
            "/aml/add",
            json={
                "request_id": "smoke:conv-9:chunk-0",
                "messages": [{"role": "user", "content": "My passport number is XK123456.", "timestamp": ts}],
                "user_id": "smoke:user-b",
                "session_id": "smoke:session:9",
            },
        )
        check("第二个用户摄入成功", r2.status_code == 200)

        def search(user_id: str, query: str, top_k: int = 10) -> list[dict]:
            r = client.post("/aml/search", json={"query": query, "user_id": user_id, "top_k": top_k})
            assert r.status_code == 200, f"search http={r.status_code}: {r.text[:200]}"
            return r.json()["data"]

        data = search("smoke:user-a", "What does the user love?")
        check("检索命中爱好相关记忆", any("hiking" in d["content"] for d in data))
        check("得分降序排列", all(data[i]["score"] >= data[i + 1]["score"] for i in range(len(data) - 1)))
        check(
            "结果项含 id/content/score/created_at",
            data and all(set(d) >= {"id", "content", "score", "created_at"} for d in data),
        )

        data = search("smoke:user-a", "passport number")
        check("user_id 隔离（不泄露他人护照号）", all("XK123456" not in d["content"] for d in data))

        data_b = search("smoke:user-b", "passport number")
        check("本人可检索到自己的记忆", any("XK123456" in d["content"] for d in data_b))

        retry = client.post("/aml/add", json=payload)
        data = search("smoke:user-a", "moved to Porto", top_k=100)
        moved = [d for d in data if "moved to Porto" in d["content"]]
        check("request_id 幂等重试不产生重复", retry.status_code == 200 and len(moved) == 1, f"hits={len(moved)}")

        noauth = httpx.Client(base_url=base, timeout=30.0)
        r = noauth.post("/aml/search", json={"query": "x", "user_id": "u", "top_k": 1})
        expected = 401 if key else 200
        check("认证按配置生效", r.status_code == expected, f"http={r.status_code}, expected={expected}")
        noauth.close()

    print()
    if failures:
        print(f"{failures} 项检查失败")
        return 1
    print("全部冒烟检查通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
