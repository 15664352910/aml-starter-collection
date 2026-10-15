"""Cross-channel fusion tests — the part of this system that is ours.

mem0 is stubbed here, so these run with no API key and no network. What they pin
down is the fusion contract: both channels reach the output, near-duplicates
collapse, extraction failure on write is retryable rather than silently lost, and
extraction failure on read degrades instead of breaking the response shape.

    EMBED_ENABLED=false python -m pytest tests/test_fusion.py -q
"""

from __future__ import annotations

import os

os.environ.setdefault("EMBED_ENABLED", "false")
os.environ.setdefault("AUTH_MODE", "none")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src import app as app_module  # noqa: E402
from src import config, store  # noqa: E402
from src.mem0_backend import Mem0Unavailable  # noqa: E402
from src.models import AddRequest, Message, SearchRequest  # noqa: E402

USER = "eval:fusion:locomo:conv-0"
SESSION = "eval:fusion:sample:0"


class StubMem0:
    """Stand-in for the mem0 core: records writes, returns canned facts."""

    def __init__(self, *, enabled: bool = True, facts: list[dict] | None = None) -> None:
        self.enabled = enabled
        self.facts = facts if facts is not None else []
        self.writes: list[tuple[str, int | None, int]] = []
        self.add_error: Exception | None = None
        self.search_error: Exception | None = None

    def add(self, messages, user_id, timestamp):  # noqa: ANN001
        if self.add_error:
            raise self.add_error
        self.writes.append((user_id, timestamp, len(messages)))

    def search(self, query, user_id, top_k):  # noqa: ANN001, ARG002
        if self.search_error:
            return []
        return self.facts[:top_k]


def fact(identifier: str, content: str, score: float = 0.9) -> dict:
    return {"id": identifier, "content": content, "score": score, "created_at": None}


def make_add(request_id: str, texts: list[str], user_id: str = USER) -> AddRequest:
    return AddRequest(
        request_id=request_id,
        user_id=user_id,
        session_id=SESSION,
        messages=[
            Message(role="user", timestamp=1704067200000 + n * 60000, content=text)
            for n, text in enumerate(texts)
        ],
    )


@pytest.fixture()
def fresh() -> store.MemoryStore:
    instance = store.MemoryStore()
    store.STORE = instance
    app_module.STORE = instance
    return instance


@pytest.fixture()
def stub(monkeypatch: pytest.MonkeyPatch) -> StubMem0:
    replacement = StubMem0()
    monkeypatch.setattr(store, "MEM0", replacement)
    monkeypatch.setattr(app_module, "STORE", store.STORE, raising=False)
    return replacement


# ── write path ──────────────────────────────────────────────────────────────


def test_add_writes_to_both_channels(fresh: store.MemoryStore, stub: StubMem0) -> None:
    written = fresh.add(make_add("req-1", ["Freya is training for the Berlin marathon."]))
    assert written == 1, "raw channel got the unit"
    assert len(stub.writes) == 1, "extraction channel was invoked"
    assert stub.writes[0][0] == USER


def test_add_passes_earliest_timestamp_to_extraction(fresh: store.MemoryStore, stub: StubMem0) -> None:
    """mem0 takes one timestamp per call; the earliest anchors relative dates.

    The store hands over milliseconds as received; converting to mem0's expected
    unit is the backend's job (covered below).
    """
    fresh.add(make_add("req-1", ["first message here", "second message here", "third one here"]))
    _, timestamp, count = stub.writes[0]
    assert timestamp == 1704067200000, "expected the earliest timestamp, unmodified"
    assert count == 3


def test_backend_converts_milliseconds_to_seconds() -> None:
    """mem0 expects seconds; the contract delivers Unix milliseconds."""
    from src.mem0_backend import Mem0Backend

    captured: dict = {}

    class FakeMemory:
        def add(self, messages, **kwargs):  # noqa: ANN001, ARG002
            captured.update(kwargs)

    backend = Mem0Backend()
    backend._memory = FakeMemory()  # noqa: SLF001
    backend.add([{"role": "user", "content": "x"}], user_id="u", timestamp=1704067200000)
    assert captured["timestamp"] == 1704067200
    assert captured["user_id"] == "u"
    assert captured["infer"] is True

    # A seconds-precision input must pass through unscaled rather than be divided.
    captured.clear()
    backend.add([{"role": "user", "content": "x"}], user_id="u", timestamp=1704067200)
    assert captured["timestamp"] == 1704067200

    # No timestamp at all means the key is simply absent.
    captured.clear()
    backend.add([{"role": "user", "content": "x"}], user_id="u", timestamp=None)
    assert "timestamp" not in captured


def test_extraction_failure_makes_add_retryable(fresh: store.MemoryStore, stub: StubMem0) -> None:
    """A failed write must not report success — the platform retries on 5xx."""
    stub.add_error = Mem0Unavailable("upstream 429")
    with pytest.raises(Mem0Unavailable):
        fresh.add(make_add("req-1", ["A memory that must not be silently dropped."]))


def test_failed_add_is_not_marked_applied(fresh: store.MemoryStore, stub: StubMem0) -> None:
    """After a failure, the platform's retry has to actually do the work."""
    stub.add_error = Mem0Unavailable("upstream 429")
    with pytest.raises(Mem0Unavailable):
        fresh.add(make_add("req-1", ["Recoverable memory content here."]))

    stub.add_error = None
    written = fresh.add(make_add("req-1", ["Recoverable memory content here."]))
    assert written == 1, "retry after failure was skipped as a duplicate"


def test_add_returns_503_so_platform_retries(fresh: store.MemoryStore, stub: StubMem0) -> None:
    stub.add_error = Mem0Unavailable("upstream 429")
    client = TestClient(app_module.app, raise_server_exceptions=False)
    response = client.post(
        "/add",
        json={
            "request_id": "req-1",
            "messages": [{"role": "user", "content": "text", "timestamp": 1704067200000}],
            "user_id": USER,
            "session_id": SESSION,
        },
    )
    assert response.status_code == 503, "503 is in the Add retry set; 200 would lose the write"


def test_tolerant_mode_keeps_add_succeeding(
    fresh: store.MemoryStore, stub: StubMem0, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "MEM0_FAIL_ADD_ON_ERROR", False)
    stub.add_error = Mem0Unavailable("upstream 429")
    assert fresh.add(make_add("req-1", ["Still indexed in the raw channel."])) == 1


# ── read path ───────────────────────────────────────────────────────────────


def test_both_channels_reach_the_output(fresh: store.MemoryStore, stub: StubMem0) -> None:
    fresh.add(make_add("req-1", ["We drove to the coast and the engine overheated badly."]))
    stub.facts = [fact("f1", "The user's car overheated on a trip to the coast.")]

    items = fresh.search(SearchRequest(query="car overheating coast", user_id=USER, top_k=10))
    identifiers = {item.id for item in items}
    assert any(i.startswith("mem_") for i in identifiers), "raw channel missing from results"
    assert "f1" in identifiers, "extracted channel missing from results"


def test_near_duplicate_across_channels_is_collapsed(fresh: store.MemoryStore, stub: StubMem0) -> None:
    """An extracted fact restating a raw turn would otherwise read to the answer
    model as independent corroboration, and waste a top_k slot."""
    text = "Miso is a tabby cat adopted last spring."
    fresh.add(make_add("req-1", [text]))
    raw = fresh.search(SearchRequest(query="Miso cat", user_id=USER, top_k=10))[0].content
    # Same content, differing only in punctuation and case.
    stub.facts = [fact("f1", raw.upper().replace(".", " !"))]

    items = fresh.search(SearchRequest(query="Miso cat", user_id=USER, top_k=10))
    assert len(items) == 1, f"duplicate survived fusion: {[i.content for i in items]}"


def test_extraction_only_still_returns_valid_items(fresh: store.MemoryStore, stub: StubMem0) -> None:
    """No raw units for this user, but the extracted channel can still answer."""
    stub.facts = [fact("f1", "The user is allergic to shellfish."), fact("f2", "Prefers window seats.")]
    items = fresh.search(SearchRequest(query="allergies", user_id="eval:fusion:empty", top_k=10))
    assert [item.id for item in items] == ["f1", "f2"]
    for item in items:
        assert item.id and item.content


def test_search_degrades_when_extraction_fails(fresh: store.MemoryStore, stub: StubMem0) -> None:
    """A malformed Search response fails the whole stage, so read-side failure
    must degrade to the raw channel rather than propagate."""
    fresh.add(make_add("req-1", ["The recipe calls for two teaspoons of smoked paprika."]))
    stub.search_error = RuntimeError("mem0 exploded")

    items = fresh.search(SearchRequest(query="smoked paprika recipe", user_id=USER, top_k=10))
    assert items, "raw channel should still answer when extraction is unavailable"
    assert all(item.id.startswith("mem_") for item in items)


def test_top_k_respected_across_fused_channels(fresh: store.MemoryStore, stub: StubMem0) -> None:
    fresh.add(make_add("req-1", [f"Raw note number {n} about hiking and maps." for n in range(15)]))
    stub.facts = [fact(f"f{n}", f"Extracted fact number {n} about hiking routes.") for n in range(15)]

    for top_k in (1, 4, 9):
        items = fresh.search(SearchRequest(query="hiking maps routes", user_id=USER, top_k=top_k))
        assert len(items) <= top_k


def test_scores_remain_non_increasing_after_fusion(fresh: store.MemoryStore, stub: StubMem0) -> None:
    fresh.add(make_add("req-1", [f"Note {n}: gardening, tomatoes, compost." for n in range(10)]))
    stub.facts = [fact(f"f{n}", f"The user grows tomatoes, batch {n}.") for n in range(10)]

    items = fresh.search(SearchRequest(query="tomatoes compost", user_id=USER, top_k=20))
    scores = [item.score for item in items]
    assert scores == sorted(scores, reverse=True)


def test_fusion_never_crosses_user_id(fresh: store.MemoryStore, stub: StubMem0) -> None:
    fresh.add(make_add("req-a", ["Alice keeps bees in Vermont."], USER))
    fresh.add(make_add("req-b", ["Bob repairs pianos in Kyoto."], "eval:fusion:other"))
    stub.facts = []

    items = fresh.search(SearchRequest(query="bees Vermont pianos Kyoto", user_id=USER, top_k=100))
    blob = " ".join(item.content for item in items)
    assert "Alice" in blob and "Bob" not in blob


def test_contract_shape_holds_with_extraction_on(fresh: store.MemoryStore, stub: StubMem0) -> None:
    """The response invariants the platform enforces must survive fusion."""
    fresh.add(make_add("req-1", ["A raw conversational turn worth remembering."]))
    stub.facts = [fact("f1", "A distilled fact about the user.")]

    client = TestClient(app_module.app)
    body = client.post(
        "/search", json={"query": "remembering facts", "user_id": USER, "top_k": 100}
    ).json()

    assert isinstance(body, dict) and "items" not in body
    assert isinstance(body["data"], list) and body["data"]
    for item in body["data"]:
        assert isinstance(item["id"], str) and item["id"]
        assert isinstance(item["content"], str) and item["content"]
