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
