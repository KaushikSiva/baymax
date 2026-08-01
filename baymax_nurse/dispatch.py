from __future__ import annotations

import json
import logging
import threading
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


logger = logging.getLogger(__name__)


class DummyDispatchReceiver:
    """Small local endpoint that proves the demo emits real HTTP POSTs."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8091) -> None:
        self.host = host
        self.port = port
        self.records: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/api/dispatch"

    def start(self) -> None:
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                if self.path != "/api/dispatch":
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                try:
                    length = min(int(self.headers.get("Content-Length", "0")), 1_000_000)
                    value = json.loads(self.rfile.read(length))
                    if not isinstance(value, dict) or not value.get("incidentId"):
                        raise ValueError("incidentId is required")
                except (ValueError, json.JSONDecodeError):
                    self.send_error(HTTPStatus.BAD_REQUEST)
                    return
                with receiver._lock:
                    receiver.records.append(value)
                    dispatch_number = len(receiver.records)
                response = json.dumps(
                    {
                        "accepted": True,
                        "dispatchId": f"dummy-dispatch-{dispatch_number:03d}",
                        "incidentId": value["incidentId"],
                    }
                ).encode("utf-8")
                self.send_response(HTTPStatus.ACCEPTED)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, format: str, *args: object) -> None:
                logger.debug("Hospital dispatch API: " + format, *args)

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="hospital-dummy-dispatch",
        )
        self._thread.start()

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=3)


class HospitalDispatchClient:
    def __init__(self, endpoint: str, *, timeout_s: float = 3.0, retries: int = 2) -> None:
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.retries = max(0, retries)

    def post(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    value = json.loads(response.read(1_000_001))
                    return {
                        "ok": True,
                        "status": response.status,
                        "attempts": attempt + 1,
                        "response": value,
                    }
            except (urllib.error.URLError, json.JSONDecodeError) as exc:
                error = exc
        return {
            "ok": False,
            "status": None,
            "attempts": self.retries + 1,
            "error": str(error),
        }
