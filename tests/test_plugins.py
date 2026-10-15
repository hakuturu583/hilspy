"""Transports and codecs written in Python."""

from __future__ import annotations

import json
import queue
from typing import Any

import pytest

import hilspy
from hilspy import Message

_queues: dict[str, queue.Queue[bytes]] = {}


class LoopTransport:
    """Everything sent comes back (a Python transport)."""

    def __init__(self, address: str, options: dict[str, str]) -> None:
        self.q = _queues.setdefault(address, queue.Queue())
        self.prefix = options.get("prefix", "").encode()
        self.closed = False

    def send(self, packet: Message) -> None:
        self.q.put(self.prefix + (packet.data or b""))

    def recv(self, timeout: float | None) -> Message | None:
        try:
            return Message(data=self.q.get(timeout=timeout))
        except queue.Empty:
            return None

    def close(self) -> None:
        self.closed = True


class JsonCodec:
    """Messages as JSON lines (a Python codec)."""

    def __init__(self, argument: str | None, options: dict[str, Any]) -> None:
        self.indent = options.get("indent")

    def encode(self, message: Message) -> list[Message]:
        doc = {"name": message.name, "fields": message.fields}
        return [Message(data=json.dumps(doc, indent=self.indent).encode())]

    def decode(self, packet: Message) -> list[Message]:
        doc = json.loads(packet.data or b"")
        if doc["name"] == "boom":
            raise ZeroDivisionError("decoder exploded")
        return [Message(doc["name"], doc["fields"])]


hilspy.register_transport("loop", LoopTransport)
hilspy.register_codec("json", JsonCodec)


def test_python_transport_with_builtin_codec() -> None:
    with hilspy.open("loop://a", "sbf") as ch:
        ch.send("AttEuler", {"heading": 1.0})
        m = ch.recv(timeout=1)
        assert m is not None and m.name == "AttEuler" and m["heading"] == 1.0


def test_python_codec_with_builtin_transport(mem: str) -> None:
    with hilspy.open(mem, "json") as tx, hilspy.open(mem, "json") as rx:
        tx.send("Hello", {"x": 1, "nested": {"y": [1, 2][0]}})
        m = rx.recv(timeout=1)
        assert m is not None and m.name == "Hello" and m.fields == {"x": 1, "nested": {"y": 1}}
        assert m.data is not None  # the packet it came from


def test_both_in_python_and_periodic() -> None:
    with hilspy.open("loop://b", "json", indent=None) as ch:
        with ch.every(0.01, "Tick", {"n": 1}) as task:
            m = ch.recv(timeout=5)
            assert m is not None and m.name == "Tick"
        assert task.sent_count >= 1


def test_plugin_exceptions_propagate() -> None:
    with hilspy.open("loop://c", "json") as ch:
        ch.send("boom", {})
        with pytest.raises(ZeroDivisionError, match="exploded"):
            ch.recv(timeout=1)


def test_url_options_reach_the_plugin() -> None:
    with hilspy.open("loop://d?prefix=", "raw") as ch:
        ch.send(Message(data=b"x"))
        m = ch.recv(timeout=1)
        assert m is not None and m.data == b"x"


def test_dropping_a_periodic_with_a_python_codec_does_not_hang() -> None:
    import gc

    with hilspy.open("loop://e", "json") as ch:
        task = ch.every(0.001, "Tick", {"n": 1})
        assert ch.recv(timeout=5) is not None
        del task
        gc.collect()


class SlowCodec(JsonCodec):
    """Encoding releases and re-takes the GIL (as Python code does)."""

    def encode(self, message: Message) -> list[Message]:
        import time

        time.sleep(0.02)
        return super().encode(message)


hilspy.register_codec("slowjson", SlowCodec)


def test_stop_while_another_thread_queries_the_task() -> None:
    # Regression: stop() held a lock while joining a sender thread that was
    # waiting for the GIL, and a second thread blocked on that lock with the
    # GIL held.
    import threading
    import time

    with hilspy.open("loop://f", "slowjson") as ch:
        task = ch.every(0.001, "Tick", {"n": 1})
        done = threading.Event()

        def poll() -> None:
            while not done.is_set():
                _ = task.running, task.period, task.sent_count, repr(task)

        poller = threading.Thread(target=poll)
        poller.start()
        time.sleep(0.05)
        stopper = threading.Thread(target=task.stop)
        stopper.start()
        stopper.join(timeout=10)
        done.set()
        poller.join(timeout=10)
        assert not stopper.is_alive() and not poller.is_alive(), "deadlock"
        assert not task.running
