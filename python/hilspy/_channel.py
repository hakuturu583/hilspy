"""Channel (sync and asyncio) and merge, over the Rust `_Channel`.

Asynchronous reception rides on `_Channel.subscribe`: a Rust thread per
channel receives and hands each message to the consumer (an asyncio queue
fed through `loop.call_soon_threadsafe`, or a thread-safe queue for `merge`
used synchronously). Awaiting never blocks the event loop, and every bus is
received concurrently, so one slow bus does not delay the others.
"""

from __future__ import annotations

import asyncio
import atexit
import queue
import sys
import threading
import weakref
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from types import TracebackType
from typing import TYPE_CHECKING, Any

from hilspy._hilspy import Error, Message, Periodic, _Channel, _open, _Subscription

if TYPE_CHECKING:
    # Annotations only (postponed by `from __future__ import annotations`):
    # typing_extensions is not a runtime dependency.
    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

Value = Any
#: How long a blocking wait sleeps between checks for Ctrl-C.
_POLL = 0.1
_live: weakref.WeakSet[Channel] = weakref.WeakSet()
#: Put into a merge's queue by close() to wake its iterator.
_WAKE: Any = object()


@atexit.register
def _stop_all() -> None:
    # Stop receive threads before the interpreter goes away under them.
    for ch in list(_live):
        ch._release()


def _raise_if_error(item: object) -> None:
    if isinstance(item, BaseException):
        raise item


class Channel:
    """A transport and a codec composed: send and receive messages.

    Synchronous: `send`, `recv`, `for msg in channel`. Asynchronous: `asend`,
    `arecv`, `async for msg in channel`. A channel is consumed one way at a
    time: once received asynchronously (or by `merge`), `recv` refuses.
    """

    __slots__ = ("__weakref__", "_inner", "_sub", "_async", "_ended", "_lock")

    def __init__(self, inner: _Channel) -> None:
        self._inner = inner
        self._sub: _Subscription | None = None
        self._async: tuple[asyncio.AbstractEventLoop, asyncio.Queue[object]] | None = None
        self._ended = False
        self._lock = threading.Lock()
        _live.add(self)

    # --- consumption by a subscription (async, merge) ---------------------

    def _subscribe(self, deliver: Callable[[object], bool]) -> None:
        with self._lock:
            if self._sub is not None:
                raise Error("the channel is already being received asynchronously or by merge()")
            self._sub = self._inner.subscribe(deliver)

    def _release(self) -> None:
        with self._lock:
            sub, self._sub, self._async = self._sub, None, None
        if sub is not None:
            sub.stop()

    def _async_queue(self) -> asyncio.Queue[object]:
        loop = asyncio.get_running_loop()
        if self._async is None:
            q: asyncio.Queue[object] = asyncio.Queue()

            def deliver(item: object) -> bool:
                try:
                    loop.call_soon_threadsafe(q.put_nowait, item)
                except RuntimeError:  # the event loop is closed
                    return False
                return item is not None

            self._subscribe(deliver)
            self._async = (loop, q)
        elif self._async[0] is not loop:
            raise Error("the channel is being received in another event loop")
        return self._async[1]

    # --- synchronous ------------------------------------------------------

    def send(self, message: Message | str, fields: Mapping[str, Value] | None = None) -> None:
        """Send a message: a `Message`, or a name and its fields."""
        self._inner.send(message, fields)

    def recv(self, timeout: float | None = None) -> Message | None:
        """The next message; None once `timeout` seconds pass."""
        if self._sub is not None:
            raise Error("the channel is being received asynchronously or by merge(); use that")
        return self._inner.recv(timeout)

    def every(
        self,
        period: float,
        message: Message | str,
        fields: Mapping[str, Value] | None = None,
        *,
        duration: float | None = None,
    ) -> Periodic:
        """Send a message every `period` seconds from a background thread."""
        return self._inner.every(period, message, fields, duration=duration)

    def close(self) -> None:
        """Close the channel: receivers (sync, async, merge) end."""
        self._inner.close()

    @property
    def closed(self) -> bool:
        return self._inner.closed

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
        self._release()

    def __iter__(self) -> Iterator[Message]:
        if self._sub is not None:
            raise Error("the channel is being received asynchronously or by merge(); use that")
        return iter(self._inner)

    # --- asynchronous -----------------------------------------------------

    async def asend(
        self, message: Message | str, fields: Mapping[str, Value] | None = None
    ) -> None:
        """Send without blocking the event loop."""
        await asyncio.to_thread(self._inner.send, message, fields)

    async def arecv(self, timeout: float | None = None) -> Message | None:
        """The next message; None once `timeout` seconds pass."""
        if self._ended:
            raise Error("the channel is closed")
        q = self._async_queue()
        try:
            if timeout is not None and timeout <= 0:
                # wait_for(..., 0) would cancel the get before it runs.
                item = q.get_nowait()
            else:
                item = await (q.get() if timeout is None else asyncio.wait_for(q.get(), timeout))
        except (asyncio.TimeoutError, asyncio.QueueEmpty):
            return None
        if item is None:
            self._ended = True
            raise Error("the channel is closed")
        _raise_if_error(item)
        assert isinstance(item, Message)
        return item

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
        await asyncio.to_thread(self._release)

    def __aiter__(self) -> AsyncIterator[Message]:
        return self._aiter()

    async def _aiter(self) -> AsyncIterator[Message]:
        while True:
            try:
                msg = await self.arecv()
            except Error:
                if self._ended:
                    return
                raise
            if msg is not None:
                yield msg

    def __repr__(self) -> str:
        return repr(self._inner).replace("_Channel", "Channel", 1)


class merge:
    """Receive from several channels at once, as they arrive.

    Yields `(channel, message)` pairs, synchronously (`for`) or with asyncio
    (`async for`), until every channel has closed; `close()` (or leaving the
    `with` block) stops early and leaves the channels open::

        async with hilspy.merge(can0, can1, gnss) as buses:
            async for bus, msg in buses:
                ...
    """

    def __init__(self, *channels: Channel) -> None:
        if not channels:
            raise ValueError("merge() needs at least one channel")
        self._channels = channels
        self._open = len(channels)
        self._sync: queue.Queue[tuple[Channel, object]] | None = None
        self._async: asyncio.Queue[tuple[Channel, object]] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closed = False

    def _start(self, put: Callable[[tuple[Channel, object]], object]) -> None:
        started: list[Channel] = []
        try:
            for ch in self._channels:

                def deliver(item: object, ch: Channel = ch) -> bool:
                    try:
                        put((ch, item))
                    except RuntimeError:  # the event loop is closed
                        return False
                    return item is not None

                if ch.closed:
                    self._open -= 1  # nothing more will come from it
                    continue
                ch._subscribe(deliver)
                started.append(ch)
        except BaseException:
            for ch in started:
                ch._release()
            raise

    def _handle(self, ch: Channel, item: object) -> tuple[Channel, Message] | None:
        if item is None:
            self._open -= 1
            return None
        _raise_if_error(item)
        assert isinstance(item, Message)
        return ch, item

    def close(self) -> None:
        """Stop receiving; the channels stay open. Iteration in progress
        (another thread or task) ends."""
        self._closed = True
        for ch in self._channels:
            ch._release()
        # Wake a waiting iterator.
        if self._sync is not None:
            self._sync.put((_WAKE, None))
        if self._async is not None and self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(self._async.put_nowait, (_WAKE, None))
            except RuntimeError:  # the event loop is closed
                pass

    def __del__(self) -> None:
        # The channels may outlive this merge: stop feeding its queue.
        try:
            self.close()
        except Exception:
            pass

    # --- synchronous ------------------------------------------------------

    def __iter__(self) -> Iterator[tuple[Channel, Message]]:
        if self._closed:
            return
        if self._sync is None:
            q: queue.Queue[tuple[Channel, object]] = queue.Queue()
            self._start(q.put)
            self._sync = q
        sync = self._sync
        while self._open > 0 and not self._closed:
            try:
                ch, item = sync.get(timeout=_POLL)  # a timeout keeps Ctrl-C working
            except queue.Empty:
                continue
            if ch is _WAKE:
                return
            pair = self._handle(ch, item)
            if pair is not None:
                yield pair

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # --- asynchronous -----------------------------------------------------

    def __aiter__(self) -> AsyncIterator[tuple[Channel, Message]]:
        return self._aiter()

    async def _aiter(self) -> AsyncIterator[tuple[Channel, Message]]:
        if self._closed:
            return
        if self._async is None:
            loop = asyncio.get_running_loop()
            q: asyncio.Queue[tuple[Channel, object]] = asyncio.Queue()
            self._start(lambda pair: loop.call_soon_threadsafe(q.put_nowait, pair))
            self._async, self._loop = q, loop
        aq = self._async
        while self._open > 0 and not self._closed:
            ch, item = await aq.get()
            if ch is _WAKE:
                return
            pair = self._handle(ch, item)
            if pair is not None:
                yield pair

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await asyncio.to_thread(self.close)


def open(url: str, codec: str = "raw", **options: Value) -> Channel:
    """Open a channel: the transport `url` with the codec `codec`.

    Transports: ``can://vcan0`` (Linux SocketCAN; ``?fd``, ``?echo``,
    ``?filter=0x100:0x700``), ``udp://host:port`` (``?listen``, ``?bind=``),
    ``tcp://host:port`` (``?listen``), ``mem://name`` (in-process),
    ``file://path`` (``?write``). Codecs: ``raw``, ``dbc:FILE`` (options
    ``strict``, ``scaling``, ``decode_choices``), ``sbf``. Keyword arguments
    configure the codec.
    """
    return Channel(_open(url, codec, **options))
