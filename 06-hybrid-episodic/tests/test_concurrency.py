"""Concurrency safety.

The platform drives Add with 64 workers and Search with 32. A lost write or a
torn read here would not raise — it would silently cost recall on a run we only
get to make once per three months. So these tests assert exact counts.

    EMBED_ENABLED=false python -m pytest tests/test_concurrency.py -q
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault("EMBED_ENABLED", "false")
os.environ.setdefault("AUTH_MODE", "none")

import pytest  # noqa: E402

from src.models import AddRequest, Message, SearchRequest  # noqa: E402
from src.store import MemoryStore  # noqa: E402

ADD_WORKERS = 64
SEARCH_WORKERS = 32


def make_add(user_id: str, chunk: int, messages: int = 8) -> AddRequest:
    return AddRequest(
        request_id=f"eval:conc:{user_id}:chunk-{chunk}",
        user_id=user_id,
        session_id=f"eval:conc:{user_id}:sample-{chunk}",
        messages=[
            Message(
                role="user" if n % 2 == 0 else "assistant",
                timestamp=1704067200000 + (chunk * 1000 + n) * 60000,
                content=(
                    f"User {user_id} chunk {chunk} message {n}: "
                    "a sufficiently long sentence about travel, food, books and weather."
                ),
            )
            for n in range(messages)
        ],
    )


@pytest.fixture()
def store() -> MemoryStore:
    return MemoryStore()


def test_concurrent_adds_lose_nothing(store: MemoryStore) -> None:
    """64 parallel Adds across 16 users must produce every unit exactly once."""
    users = [f"eval:conc:u{n}" for n in range(16)]
    jobs = [(user, chunk) for user in users for chunk in range(4)]

    with ThreadPoolExecutor(max_workers=ADD_WORKERS) as pool:
        futures = [pool.submit(store.add, make_add(user, chunk)) for user, chunk in jobs]
        written = [future.result() for future in as_completed(futures)]

    expected_per_chunk = written[0]
    assert expected_per_chunk > 0
    assert all(count == expected_per_chunk for count in written), "uneven unit counts"

    stats = store.stats()
    assert stats["users"] == len(users)
    assert stats["memories"] == len(jobs) * expected_per_chunk, (
        f"lost writes: expected {len(jobs) * expected_per_chunk}, got {stats['memories']}"
    )


def test_ids_stay_unique_under_concurrent_adds(store: MemoryStore) -> None:
    """The per-user id counter must not hand out a duplicate under contention."""
    user = "eval:conc:single"
    with ThreadPoolExecutor(max_workers=ADD_WORKERS) as pool:
        list(
            as_completed(
                [pool.submit(store.add, make_add(user, chunk)) for chunk in range(ADD_WORKERS)]
            )
        )

    # Check the whole index, not a Search result: MAX_RETURN caps what Search
    # emits, so the returned page is a subset by design.
    index = store._index_for(user)  # noqa: SLF001 - asserting an internal invariant
    assert len(index.ids) == len(set(index.ids)), "duplicate memory ids under concurrency"
    assert len(index.ids) == store.stats()["memories"]

    # The emitted page must also be duplicate-free and respect the cap.
    returned = store.search(SearchRequest(query="travel food books weather", user_id=user, top_k=100_000))
    ids = [item.id for item in returned]
    assert len(ids) == len(set(ids))
    assert len(ids) <= min(100_000, len(index.ids))


def test_concurrent_add_and_search_interleaved(store: MemoryStore) -> None:
    """Searching while writes land must never raise or return a malformed item."""
    users = [f"eval:conc:mixed{n}" for n in range(8)]
    errors: list[BaseException] = []

    def writer(user: str, chunk: int) -> None:
        store.add(make_add(user, chunk))

    def reader(user: str) -> None:
        for item in store.search(SearchRequest(query="travel books", user_id=user, top_k=50)):
            # Every emitted item must satisfy the contract even mid-write.
            assert isinstance(item.id, str) and item.id
            assert isinstance(item.content, str) and item.content

    with ThreadPoolExecutor(max_workers=ADD_WORKERS + SEARCH_WORKERS) as pool:
        futures = []
        for chunk in range(6):
            for user in users:
                futures.append(pool.submit(writer, user, chunk))
                futures.append(pool.submit(reader, user))
        for future in as_completed(futures):
            try:
                future.result()
            except BaseException as error:  # noqa: BLE001
                errors.append(error)

    assert not errors, f"{len(errors)} failure(s) during interleaved add/search: {errors[:3]}"


def test_retried_adds_under_concurrency_stay_idempotent(store: MemoryStore) -> None:
    """The same request_id fired 64 ways at once must index exactly one copy."""
    user = "eval:conc:retry"
    request = make_add(user, 0)

    with ThreadPoolExecutor(max_workers=ADD_WORKERS) as pool:
        counts = [
            future.result()
            for future in as_completed(
                [pool.submit(store.add, request) for _ in range(ADD_WORKERS)]
            )
        ]

    applied = [count for count in counts if count > 0]
    assert len(applied) == 1, f"request_id applied {len(applied)} times, expected exactly 1"
    assert store.stats()["memories"] == applied[0]


def test_idempotency_holds_when_the_write_is_slow(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: a retry arriving *during* the original write.

    The platform's Add timeout is its own, so a retry can land while the first
    call is still embedding. This originally slipped through — ownership was
    claimed only after the slow part, so every thread passed the duplicate check
    first. It reproduced solely because the real encoder was slow enough; forcing
    the delay makes it deterministic regardless of backend speed.
    """
    from src import store as store_module

    real_encode = store_module.EMBEDDER.encode

    def slow_encode(texts, *, is_query=False):  # noqa: ANN001, ANN202
        if not is_query:
            time.sleep(0.25)
        return real_encode(texts, is_query=is_query)

    monkeypatch.setattr(store_module.EMBEDDER, "encode", slow_encode)

    user = "eval:conc:slow-retry"
    request = make_add(user, 0)
    with ThreadPoolExecutor(max_workers=ADD_WORKERS) as pool:
        counts = [
            future.result()
            for future in as_completed(
                [pool.submit(store.add, request) for _ in range(ADD_WORKERS)]
            )
        ]

    applied = [count for count in counts if count > 0]
    assert len(applied) == 1, (
        f"a retry overlapping a slow write double-indexed: applied {len(applied)} times"
    )
    index = store._index_for(user)  # noqa: SLF001
    assert len(index.ids) == applied[0]
    assert len(index.ids) == len(set(index.ids))


def test_duplicate_waits_so_200_still_means_retrievable(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A duplicate must not return before the owner's write is searchable.

    Returning 200 early would break the contract's core guarantee for that
    response, even though the owner eventually lands the data.
    """
    from src import store as store_module

    real_encode = store_module.EMBEDDER.encode

    def slow_encode(texts, *, is_query=False):  # noqa: ANN001, ANN202
        if not is_query:
            time.sleep(0.3)
        return real_encode(texts, is_query=is_query)

    monkeypatch.setattr(store_module.EMBEDDER, "encode", slow_encode)

    user = "eval:conc:wait"
    request = make_add(user, 0)

    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(store.add, request)
        time.sleep(0.05)  # let the owner claim it and start the slow part
        duplicate = pool.submit(store.add, request)
        assert duplicate.result() == 0
        # The duplicate returned only after the owner finished, so the memory is
        # already queryable at that point.
        assert store._index_for(user).ids, "duplicate returned before data landed"  # noqa: SLF001
        owner.result()


def test_failed_owner_makes_waiting_duplicate_raise(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the owner fails, a waiting duplicate must not report success."""
    from src import store as store_module

    def exploding_encode(texts, *, is_query=False):  # noqa: ANN001, ANN202
        if not is_query:
            time.sleep(0.2)
            raise RuntimeError("encoder died")
        return None

    monkeypatch.setattr(store_module.EMBEDDER, "encode", exploding_encode)

    user = "eval:conc:owner-fails"
    request = make_add(user, 0)

    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(store.add, request)
        time.sleep(0.05)
        duplicate = pool.submit(store.add, request)

        with pytest.raises(RuntimeError):
            owner.result()
        with pytest.raises(BaseException):
            duplicate.result()

    # Nothing was marked complete, so the platform's retry will redo the work.
    assert not store._index_for(user).is_completed(request.request_id)  # noqa: SLF001


def test_no_cross_user_leak_under_concurrency(store: MemoryStore) -> None:
    """user_id partitioning must hold while 64 writers touch 16 partitions."""
    users = [f"eval:conc:iso{n}" for n in range(16)]
    with ThreadPoolExecutor(max_workers=ADD_WORKERS) as pool:
        list(
            as_completed(
                [
                    pool.submit(store.add, make_add(user, chunk))
                    for user in users
                    for chunk in range(4)
                ]
            )
        )

    for user in users:
        for item in store.search(
            SearchRequest(query="travel food books weather", user_id=user, top_k=100)
        ):
            assert user in item.content, f"{user} received a memory belonging elsewhere"
