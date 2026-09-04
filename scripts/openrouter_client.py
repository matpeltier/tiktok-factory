"""Minimal OpenRouter chat-completions client with native video input.

Videos are sent inline as base64 data URLs (`video_url` content parts), which
works with models whose `input_modalities` include "video" (e.g.
`google/gemini-3.8-flash`). Usage and cost reported by OpenRouter are returned
so each pipeline pass can record real token usage and cost.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-3.8-flash"
DEFAULT_TIMEOUT_SECONDS = 900
DEFAULT_MAX_TOKENS = 65536
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5.0
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class OpenRouterError(RuntimeError):
    """Raised when an OpenRouter call cannot be completed."""


class OpenRouterHTTPError(OpenRouterError):
    """Raised when OpenRouter returns a non-200 HTTP response."""

    def __init__(self, status_code: int, body_text: str):
        super().__init__(f"OpenRouter HTTP {status_code}: {body_text}")
        self.status_code = status_code


@dataclass(frozen=True)
class CallResult:
    content: str
    model: str
    response_id: str
    finish_reason: str | None
    usage: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)


def load_env_file(path: Path | str) -> None:
    """Load KEY=VALUE pairs into os.environ without overriding existing values."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _api_key() -> str:
    load_env_file(Path(__file__).resolve().parent.parent / ".env")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise OpenRouterError(
            "OPENROUTER_API_KEY is not set. Export it or put it in a .env at the repo root."
        )
    return key


def video_data_url(video_path: Path | str) -> str:
    """Encode a local video file as a base64 data URL accepted by OpenRouter."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    mime = mimetypes.guess_type(video_path.name)[0] or "video/mp4"
    if not mime.startswith("video/"):
        raise OpenRouterError(f"Unsupported video mime type for {video_path.name}: {mime}")
    encoded = base64.b64encode(video_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def image_data_url_part(image_path: Path | str) -> dict:
    """Build an `image_url` content part from a local image file."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    if not mime.startswith("image/"):
        raise OpenRouterError(f"Unsupported image mime type for {image_path.name}: {mime}")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}


def parse_json_content(content: str) -> dict:
    """Parse a model response that must be a JSON object, tolerating code fences."""
    text = content.strip()
    if text.startswith("```"):
        body = text[3:]
        if body.startswith("json"):
            body = body[4:]
        if body.rstrip().endswith("```"):
            body = body.rstrip()[:-3]
        text = body.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise OpenRouterError(f"Model response is not valid JSON: {exc}\n---\n{content[:500]}") from exc
    if not isinstance(parsed, dict):
        raise OpenRouterError(f"Model response is JSON but not an object: {type(parsed).__name__}")
    return parsed


def call_openrouter(
    prompt: str,
    *,
    video_path: Path | str | None = None,
    image_paths: list[Path | str] | None = None,
    system: str | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    retries: int = MAX_RETRIES,
    json_mode: bool = False,
) -> CallResult:
    """Send one chat completion to OpenRouter, optionally with media.

    Videos are sent as a single base64 data URL (`video_url` part); images as
    `image_url` parts. Some providers enforce a higher billing minimum on
    video requests than on images.

    With ``json_mode=True`` the response is constrained to a syntactically valid
    JSON object (equivalent to Gemini's ``response_mime_type="application/json"``)
    while the prompt keeps driving the content.

    Returns a CallResult with the assistant content and the usage/cost reported
    by OpenRouter. Retries transient HTTP failures with linear backoff.
    """
    content_parts: list[dict] = []
    if video_path is not None:
        content_parts.append({"type": "video_url", "video_url": {"url": video_data_url(video_path)}})
    for image_path in image_paths or []:
        content_parts.append(image_data_url_part(image_path))
    content: list[dict] | str
    if content_parts:
        content_parts.append({"type": "text", "text": prompt})
        content = content_parts
    else:
        content = prompt

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning": {"exclude": True},
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
    }

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=timeout_seconds)
        except requests.RequestException as exc:
            last_error = exc
        else:
            if response.status_code == 200:
                try:
                    body = response.json()
                except ValueError as exc:
                    last_error = OpenRouterError(f"OpenRouter returned non-JSON 200 body: {exc}")
                else:
                    # OpenRouter sometimes reports provider errors inside a 200
                    # response (e.g. 413 payload-too-large) instead of an HTTP
                    # error status; surface them with their real code.
                    embedded = body.get("error") if isinstance(body, dict) else None
                    if embedded:
                        code = embedded.get("code") if isinstance(embedded.get("code"), int) else 500
                        error = OpenRouterHTTPError(code, str(embedded.get("message"))[:500])
                        if code in RETRYABLE_STATUS_CODES:
                            last_error = error
                        else:
                            raise error
                    else:
                        choice = (body.get("choices") or [{}])[0]
                        message = choice.get("message") or {}
                        text = message.get("content")
                        if not text:
                            raise OpenRouterError(
                                "Model returned empty content "
                                f"(finish_reason={choice.get('finish_reason')!r}). "
                                "Increase max_tokens so reasoning plus the full JSON answer fit."
                            )
                        usage = dict(body.get("usage") or {})
                        return CallResult(
                            content=text,
                            model=body.get("model", model),
                            response_id=body.get("id", ""),
                            finish_reason=choice.get("finish_reason"),
                            usage=usage,
                            raw=body,
                        )
            elif response.status_code in RETRYABLE_STATUS_CODES:
                last_error = OpenRouterHTTPError(response.status_code, response.text[:500])
            else:
                raise OpenRouterHTTPError(response.status_code, response.text[:500])
        if attempt < retries:
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    if isinstance(last_error, OpenRouterHTTPError):
        raise last_error
    raise OpenRouterError(f"OpenRouter call failed after {retries} attempts: {last_error}")


def call_openrouter_json(
    prompt: str,
    *,
    video_path: Path | str | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    retries: int = MAX_RETRIES,
) -> tuple[dict, CallResult, CallResult | None]:
    """Call OpenRouter and return parsed JSON, repairing malformed responses.

    Some providers emit structurally invalid JSON (e.g. a missing closing brace
    in long nested outputs) even in JSON mode. When parsing fails, a cheap
    text-only repair pass asks the model to fix and re-emit the complete JSON.

    Returns ``(parsed, call, repair_call_or_none)``.
    """
    call = call_openrouter(
        prompt,
        video_path=video_path,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        retries=retries,
        json_mode=True,
    )
    try:
        parsed = parse_json_content(call.content)
        return parsed, call, None
    except OpenRouterError as parse_error:
        repair_prompt = (
            "The following text was intended to be a single JSON object but contains a "
            f"syntax error: {parse_error}\n\n"
            "Fix the syntax error and return the COMPLETE corrected JSON object. "
            "Preserve every field and value exactly; only fix the JSON structure. "
            "Return ONLY the JSON object, with no surrounding text.\n\n---\n"
            f"{call.content}\n---"
        )
        repair_call = call_openrouter(
            repair_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            retries=retries,
            json_mode=True,
        )
        parsed = parse_json_content(repair_call.content)
        return parsed, call, repair_call
