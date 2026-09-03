"""Tests for the OpenRouter client used by the decompilation pipeline."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import openrouter_client as orc


class FakeResponse:
    def __init__(self, status_code=200, body=None, text=""):
        self.status_code = status_code
        self._body = body or {}
        self.text = text or json.dumps(self._body)

    def json(self):
        return self._body


def _completion_body(content='{"ok": true}', finish_reason="stop"):
    return {
        "id": "gen-test-1",
        "model": "google/gemini-3.8-flash",
        "choices": [
            {"index": 0, "finish_reason": finish_reason, "message": {"role": "assistant", "content": content}}
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "cost": 0.001,
        },
    }


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(orc, "load_env_file", lambda path: None)
    with pytest.raises(orc.OpenRouterError, match="OPENROUTER_API_KEY"):
        orc.call_openrouter("hello")


def test_video_data_url_encodes_file(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"\x00\x01\x02fakevideo")
    url = orc.video_data_url(video)
    assert url.startswith("data:video/mp4;base64,")
    import base64

    assert base64.b64decode(url.split(",", 1)[1]) == b"\x00\x01\x02fakevideo"


def test_video_data_url_rejects_missing_file():
    with pytest.raises(FileNotFoundError):
        orc.video_data_url("/nonexistent/video.mp4")


def test_call_openrouter_success_with_video(monkeypatch, tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video-bytes")
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return FakeResponse(200, _completion_body())

    monkeypatch.setattr(orc.requests, "post", fake_post)
    result = orc.call_openrouter("analyze this", video_path=video, model="google/gemini-3.8-flash")

    assert captured["url"] == orc.API_URL
    assert captured["headers"]["Authorization"] == "Bearer sk-or-test"
    payload = captured["json"]
    assert payload["model"] == "google/gemini-3.8-flash"
    parts = payload["messages"][0]["content"]
    assert parts[0]["type"] == "video_url"
    assert parts[0]["video_url"]["url"].startswith("data:video/mp4;base64,")
    assert parts[1] == {"type": "text", "text": "analyze this"}
    assert result.content == '{"ok": true}'
    assert result.usage["cost"] == 0.001
    assert result.usage["total_tokens"] == 150
    assert result.finish_reason == "stop"


def test_call_openrouter_retries_on_429(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(1)
        if len(calls) < 3:
            return FakeResponse(429, {"error": {"message": "rate limited"}})
        return FakeResponse(200, _completion_body())

    monkeypatch.setattr(orc.requests, "post", fake_post)
    monkeypatch.setattr(orc.time, "sleep", lambda seconds: None)
    result = orc.call_openrouter("hello", retries=3)
    assert len(calls) == 3
    assert result.content == '{"ok": true}'


def test_call_openrouter_raises_after_retries(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(1)
        return FakeResponse(503, {"error": {"message": "unavailable"}})

    monkeypatch.setattr(orc.requests, "post", fake_post)
    monkeypatch.setattr(orc.time, "sleep", lambda seconds: None)
    with pytest.raises(orc.OpenRouterHTTPError) as exc_info:
        orc.call_openrouter("hello", retries=3)
    assert exc_info.value.status_code == 503
    assert len(calls) == 3


def test_call_openrouter_non_retryable_error_raises_immediately(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(1)
        return FakeResponse(401, {"error": {"message": "invalid key"}})

    monkeypatch.setattr(orc.requests, "post", fake_post)
    with pytest.raises(orc.OpenRouterHTTPError, match="401"):
        orc.call_openrouter("hello", retries=3)
    assert len(calls) == 1


def test_call_openrouter_empty_content_raises(monkeypatch):
    body = _completion_body(content=None, finish_reason="length")
    monkeypatch.setattr(orc.requests, "post", lambda *a, **k: FakeResponse(200, body))
    with pytest.raises(orc.OpenRouterError, match="empty content"):
        orc.call_openrouter("hello")


def test_parse_json_content_tolerates_fences():
    assert orc.parse_json_content('```json\n{"a": 1}\n```') == {"a": 1}
    assert orc.parse_json_content('{"a": 2}') == {"a": 2}
    assert orc.parse_json_content('```\n{"a": 3}\n```') == {"a": 3}
    with pytest.raises(orc.OpenRouterError, match="not valid JSON"):
        orc.parse_json_content("not json at all")


def test_json_mode_sets_response_format(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["json"] = json
        return FakeResponse(200, _completion_body())

    monkeypatch.setattr(orc.requests, "post", fake_post)
    orc.call_openrouter("hello", json_mode=True)
    assert captured["json"]["response_format"] == {"type": "json_object"}

    orc.call_openrouter("hello")
    assert "response_format" not in captured["json"]


def test_call_openrouter_json_repair(monkeypatch):
    broken = '{"shots": [{"id": "shot_0"}, {"id": "shot_1"'
    fixed = '{"shots": [{"id": "shot_0"}, {"id": "shot_1"}]}'
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json)
        if len(calls) == 1:
            return FakeResponse(200, _completion_body(content=broken))
        return FakeResponse(200, _completion_body(content=fixed))

    monkeypatch.setattr(orc.requests, "post", fake_post)
    parsed, call, repair = orc.call_openrouter_json("analyze")
    assert len(calls) == 2
    assert parsed == {"shots": [{"id": "shot_0"}, {"id": "shot_1"}]}
    assert call.content == broken
    assert repair is not None and repair.content == fixed
    assert "syntax error" in calls[1]["messages"][0]["content"]


def test_call_openrouter_json_no_repair_needed(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json)
        return FakeResponse(200, _completion_body(content='{"ok": 1}'))

    monkeypatch.setattr(orc.requests, "post", fake_post)
    parsed, call, repair = orc.call_openrouter_json("analyze")
    assert len(calls) == 1
    assert parsed == {"ok": 1}
    assert repair is None


def test_call_openrouter_json_repair_still_broken_raises(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        return FakeResponse(200, _completion_body(content='{"still": broken'))

    monkeypatch.setattr(orc.requests, "post", fake_post)
    with pytest.raises(orc.OpenRouterError, match="not valid JSON"):
        orc.call_openrouter_json("analyze")


def test_parse_json_content_rejects_non_object():
    with pytest.raises(orc.OpenRouterError, match="not an object"):
        orc.parse_json_content("[1, 2, 3]")
    with pytest.raises(orc.OpenRouterError, match="not an object"):
        orc.parse_json_content("42")


def test_parse_json_content_fence_without_newline():
    assert orc.parse_json_content('```json{"a": 1}```') == {"a": 1}
    assert orc.parse_json_content('```{"a": 2}```') == {"a": 2}


def test_call_openrouter_non_json_200_body_retries(monkeypatch):
    calls = []

    class HtmlResponse:
        status_code = 200
        text = "<html>gateway error</html>"

        def json(self):
            raise ValueError("No JSON")

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(1)
        if len(calls) == 1:
            return HtmlResponse()
        return FakeResponse(200, _completion_body())

    monkeypatch.setattr(orc.requests, "post", fake_post)
    monkeypatch.setattr(orc.time, "sleep", lambda seconds: None)
    result = orc.call_openrouter("hello", retries=2)
    assert len(calls) == 2
    assert result.content == '{"ok": true}'


def test_call_openrouter_retry_exhaustion_preserves_http_error(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        return FakeResponse(429, {"error": {"message": "rate limited"}})

    monkeypatch.setattr(orc.requests, "post", fake_post)
    monkeypatch.setattr(orc.time, "sleep", lambda seconds: None)
    with pytest.raises(orc.OpenRouterHTTPError) as exc_info:
        orc.call_openrouter("hello", retries=2)
    assert exc_info.value.status_code == 429


def test_video_data_url_rejects_non_video_file(tmp_path):
    not_video = tmp_path / "notes.txt"
    not_video.write_text("hello")
    with pytest.raises(orc.OpenRouterError, match="Unsupported video mime"):
        orc.video_data_url(not_video)


def test_text_only_call_sends_plain_string_content(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["json"] = json
        return FakeResponse(200, _completion_body())

    monkeypatch.setattr(orc.requests, "post", fake_post)
    orc.call_openrouter("fix this json")
    assert captured["json"]["messages"][0]["content"] == "fix this json"


def test_call_openrouter_surfaces_embedded_413(monkeypatch):
    body = {"id": "x", "error": {"message": "Request body exceeds the provider maximum size", "code": 413}}
    monkeypatch.setattr(orc.requests, "post", lambda *a, **k: FakeResponse(200, body))
    with pytest.raises(orc.OpenRouterHTTPError) as exc_info:
        orc.call_openrouter("hello")
    assert exc_info.value.status_code == 413


def test_call_openrouter_embedded_retryable_error_retries(monkeypatch):
    calls = []
    body = {"id": "x", "error": {"message": "provider overloaded", "code": 503}}

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(1)
        if len(calls) < 2:
            return FakeResponse(200, body)
        return FakeResponse(200, _completion_body())

    monkeypatch.setattr(orc.requests, "post", fake_post)
    monkeypatch.setattr(orc.time, "sleep", lambda seconds: None)
    result = orc.call_openrouter("hello", retries=2)
    assert len(calls) == 2
    assert result.content == '{"ok": true}'
