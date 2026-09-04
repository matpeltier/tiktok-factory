"""Thin fal.ai queue client: submit, poll, fetch, download.

fal's queue API: POST https://queue.fal.run/{endpoint_id} returns a request_id
plus status/response URLs; results are fetched from the response URL once the
job reaches COMPLETED. fal does not return monetary cost in the response, so
costs are tracked by the caller from a per-model price table.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

QUEUE_URL = "https://queue.fal.run"
POLL_INTERVAL_SECONDS = 5.0
DEFAULT_TIMEOUT_SECONDS = 600
DOWNLOAD_CHUNK = 1 << 20


class FalError(RuntimeError):
    """Raised when a fal call cannot be completed."""


class FalHTTPError(FalError):
    def __init__(self, status_code: int, body_text: str):
        super().__init__(f"fal HTTP {status_code}: {body_text[:500]}")
        self.status_code = status_code


@dataclass(frozen=True)
class FalResult:
    request_id: str
    payload: dict
    status_url: str = ""
    response_url: str = ""
    raw: dict = field(default_factory=dict)


def _key() -> str:
    key = os.environ.get("FAL_KEY")
    if not key:
        raise FalError("FAL_KEY is not set.")
    return key


def _headers() -> dict:
    return {"Authorization": f"Key {_key()}", "Content-Type": "application/json"}


def submit(endpoint_id: str, payload: dict, timeout_seconds: int = 120) -> FalResult:
    response = requests.post(f"{QUEUE_URL}/{endpoint_id}", headers=_headers(), json=payload, timeout=timeout_seconds)
    if response.status_code not in (200, 202):
        raise FalHTTPError(response.status_code, response.text)
    body = response.json()
    request_id = body.get("request_id") or body.get("requestId")
    if not request_id:
        raise FalError(f"fal submit returned no request_id: {str(body)[:300]}")
    # fal returns ready-to-use status/response URLs whose path may differ from
    # the endpoint id (e.g. only the first path segment); never reconstruct them.
    return FalResult(
        request_id=request_id,
        payload=body,
        status_url=body.get("status_url", ""),
        response_url=body.get("response_url", ""),
        raw=body,
    )


def status(status_url: str, timeout_seconds: int = 60) -> dict:
    response = requests.get(status_url, headers=_headers(), timeout=timeout_seconds)
    # 200 = COMPLETED payload, 202 = IN_QUEUE/IN_PROGRESS accepted response
    if response.status_code not in (200, 202):
        raise FalHTTPError(response.status_code, response.text)
    return response.json()


def result(response_url: str, timeout_seconds: int = 120) -> dict:
    response = requests.get(response_url, headers=_headers(), timeout=timeout_seconds)
    if response.status_code not in (200, 202):
        raise FalHTTPError(response.status_code, response.text)
    return response.json()


def wait_for_result(
    submitted: FalResult,
    *,
    poll_interval: float = POLL_INTERVAL_SECONDS,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict:
    """Poll the queue until COMPLETED and return the final response payload."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        state = status(submitted.status_url)
        fal_status = state.get("status")
        if fal_status == "COMPLETED":
            return result(submitted.response_url)
        if fal_status not in ("IN_QUEUE", "IN_PROGRESS"):
            raise FalError(f"Unexpected fal status {fal_status!r}: {str(state)[:300]}")
        if time.monotonic() > deadline:
            raise FalError(f"fal job {submitted.request_id} timed out after {timeout_seconds}s (status={fal_status})")
        time.sleep(poll_interval)


def run(endpoint_id: str, payload: dict, *, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> dict:
    """Submit and wait for one fal job."""
    submitted = submit(endpoint_id, payload)
    return wait_for_result(submitted, timeout_seconds=timeout_seconds)


def download(url: str, destination: Path, *, chunk_size: int = DOWNLOAD_CHUNK) -> Path:
    """Stream a (typically CDN) URL into a local file."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as response:
        if response.status_code != 200:
            raise FalHTTPError(response.status_code, response.text[:300])
        with open(destination, "wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                handle.write(chunk)
    return destination
