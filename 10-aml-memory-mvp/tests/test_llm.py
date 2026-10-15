import json
from pathlib import Path
from unittest.mock import patch

from app.config import Settings
from app.llm import MemoryLLM
from app.schemas import MemoryMessage


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def llm_settings() -> Settings:
    return Settings(
        database_path=Path("unused.db"),
        api_key=None,
        llm_mode="required",
        openai_api_key="test-key",
        openai_base_url="https://api.openai.com/v1",
        openai_model="gpt-4o-mini",
        openai_timeout_seconds=1,
        candidate_limit=100,
    )


def test_add_and_search_use_configured_model_without_answering() -> None:
    captured_requests = []

    def fake_urlopen(request, timeout):
        captured_requests.append((request, timeout))
        response = {"choices": [{"message": {"content": '{"terms":["Alice","2026"]}'}}]}
        return FakeResponse(response)

    llm = MemoryLLM(llm_settings())
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        annotations = llm.annotate_messages(
            [MemoryMessage(role="user", content="Alice moved in 2026.")]
        )
        analysis = llm.analyze_query("When did Alice move?", None)

    assert annotations == ["Alice 2026"]
    assert analysis.terms == ["Alice", "2026"]
    assert analysis.facets == []
    assert analysis.temporal_intent == "none"
    assert len(captured_requests) == 2
    for request, timeout in captured_requests:
        body = json.loads(request.data.decode("utf-8"))
        assert body["model"] == "gpt-4o-mini"
        assert body["temperature"] == 0
        assert body["max_tokens"] == 256
        assert timeout == 1
        assert "answer" in body["messages"][0]["content"].lower()


def test_message_annotations_remain_attached_to_source_message() -> None:
    response = {
        "choices": [{"message": {"content": json.dumps({
            "items": [
                {"index": 0, "terms": ["Alice", "Kyoto"]},
                {"index": 1, "terms": ["Bob", "Osaka"]},
            ]
        })}}]
    }
    llm = MemoryLLM(llm_settings())
    with patch("urllib.request.urlopen", return_value=FakeResponse(response)):
        annotations = llm.annotate_messages([
            MemoryMessage(role="user", content="Alice moved to Kyoto."),
            MemoryMessage(role="assistant", content="Bob moved to Osaka."),
        ])
    assert annotations == ["Alice Kyoto", "Bob Osaka"]
