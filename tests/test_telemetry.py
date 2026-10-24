"""Metrics export over OTLP/HTTP, turned on by OTEL_EXPORTER_OTLP_ENDPOINT.

The exporter is installed once per process, so each case runs a child
process against a stand-in OTLP receiver.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

CHILD = """
import hilspy

tx = hilspy.open("mem://otel-test")
rx = hilspy.open("mem://otel-test")
for i in range(5):
    tx.send(hilspy.Message(data=bytes(8), id=0x123))
    rx.recv(timeout=1.0)
# Exported at exit (atexit), well within the interval.
"""


class Receiver(ThreadingHTTPServer):
    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), Handler)
        self.requests: list[tuple[str, str, bytes]] = []


class Handler(BaseHTTPRequestHandler):
    server: Receiver

    def do_POST(self) -> None:  # noqa: N802 (http.server's name)
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.server.requests.append((self.path, self.headers["Content-Type"], body))
        self.send_response(200)
        self.send_header("Content-Type", "application/x-protobuf")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def receiver() -> Iterator[Receiver]:
    server = Receiver()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def run_child(**env: str) -> subprocess.CompletedProcess[str]:
    base = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("OTEL_") and k.lower() not in ("http_proxy", "all_proxy")
    }
    return subprocess.run(
        [sys.executable, "-c", CHILD],
        env={**base, **env},
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_exports_bus_traffic_and_host_load(receiver: Receiver) -> None:
    host, port = receiver.server_address[:2]
    child = run_child(
        OTEL_EXPORTER_OTLP_ENDPOINT=f"http://{host!s}:{port}",
        OTEL_SERVICE_NAME="test-ecu",
        # Longer than the child lives: what arrives is the exit's export.
        OTEL_METRIC_EXPORT_INTERVAL="60000",
    )
    assert child.returncode == 0, child.stderr
    assert receiver.requests, "nothing exported"
    path, content_type, body = receiver.requests[-1]
    assert path == "/v1/metrics"
    assert content_type == "application/x-protobuf"
    # Protobuf keeps strings as they are.
    for text in (
        b"hilspy.packets",
        b"hilspy.messages",
        b"hilspy.can.bits",
        b"mem://otel-test",
        b"0x123",
        b"process.cpu.utilization",
        b"system.memory.utilization",
        b"test-ecu",
    ):
        assert text in body, text


def test_nothing_is_exported_without_an_endpoint(receiver: Receiver) -> None:
    host, port = receiver.server_address[:2]
    assert run_child().returncode == 0
    disabled = run_child(
        OTEL_EXPORTER_OTLP_ENDPOINT=f"http://{host!s}:{port}", OTEL_SDK_DISABLED="true"
    )
    assert disabled.returncode == 0, disabled.stderr
    assert receiver.requests == []


def test_an_unsupported_protocol_fails_loudly() -> None:
    child = run_child(
        OTEL_EXPORTER_OTLP_ENDPOINT="http://127.0.0.1:9", OTEL_EXPORTER_OTLP_PROTOCOL="grpc"
    )
    assert child.returncode != 0
    assert "hilspy.Error: telemetry: OTEL_EXPORTER_OTLP_PROTOCOL" in child.stderr
