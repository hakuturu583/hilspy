# Design

## Layers

```
Python API     python/hilspy/__init__.py      open, Channel, Message, Periodic, Error, register_*
PyO3 layer     crates/hilspy-python           conversions, GIL release, Python plugins as Rust traits
Rust core      crates/hilspy-core             Channel = Transport + Codec; registries; formats
Crates         socketcan, can-dbc, libsbf     I/O and parsing
```

The core has no Python dependency, so a Rust-side scenario runner or a
real-time process can use it directly.

## Channel = Transport + Codec

```
           send(Message)                               recv() -> Message
               |                                              ^
         name? |-- no --> its Packet                          | (pending queue)
               |                                              |
           Codec::encode -> [Packet]                  Codec::decode(Packet) -> [Message]
               |                                              ^
           Transport::send(Packet)                    Transport::recv(timeout) -> Packet
```

* **Packet**: bytes, an optional CAN header (`id`, `extended`, `fd`, `brs`)
  and a reception timestamp. It is what a transport moves: a CAN frame, a
  UDP datagram, a TCP chunk, a file chunk.
* **Message**: an optional name, named fields (`Value`: null, bool, int,
  float, str, map) and the packet it came from or goes out as. A nameless
  message bypasses the codec in both directions.
* **Transport** (`send`, `recv(timeout)`, `close`; `&self`, `Send + Sync`): one
  transport serves a receiving thread and periodic senders at once.
* **Codec** (`encode(&mut, &Message)`, `decode(&mut, Packet)`; `Send`): may keep
  state across packets (the SBF decoder reassembles blocks across TCP chunks).
  The channel holds it behind a mutex.
* **Periodic**: one thread per cyclic message with absolute deadlines (no
  drift; a deadline missed by more than a period is skipped). The message is
  encoded on every tick, so time-stamping codecs (`sbf` fills `tow`/`wnc`)
  stamp each send; it is encoded once up front so a bad message fails at
  `every()`, not silently on the thread.

## Pluggability

`transport::register(scheme, factory)` and `codec::register(name, factory)`
hold `Arc<dyn Fn(...)>` factories in process-wide registries that the built-in
ones also live in; `Channel::open(url, codec, options)` looks both up.
Options are typed `Value`s; URL query values are strings that readers parse
(`?fd`, `?fd=1`, `?fd=true`), and `Options::finish` rejects any option a
transport or codec did not consume, so a typo fails loudly.

The Python layer registers Python objects through the same registries:
`PyTransport` and `PyCodec` implement the Rust traits by calling the object's
methods with the GIL attached. Every core call that can reach a codec or
transport (open, send, recv, every, update) runs with the GIL released, so a
Rust thread (a periodic sender) can call into a Python plugin without
deadlocking. A Python exception raised in a plugin travels through the core as
`Error::External` and is re-raised unchanged.

## Transports

* `can://`: raw `CanSocket` or, with `?fd`, `CanFdSocket`; kernel filters
  (`?filter=ID:MASK`, the identifier kind matched through `CAN_EFF_FLAG`);
  receive polls in 100 ms slices so `close()` wakes a blocked receiver. Remote
  and error frames are not delivered.
* `mem://`: a registry of hubs by name; each endpoint has a bounded queue
  (oldest dropped) and software filters.
* `udp://`: one datagram per packet; a listener replies to the last sender.
* `tcp://`: reader threads push chunks into a queue; a server broadcasts sends
  to every connected client and drops broken ones. A client channel ends when
  the connection does.
* `file://`: replay in 64 KiB chunks (ends at EOF, or `?follow`), or record
  packet bytes (`?write`, `?append`).

## Codecs

* `dbc`: `can-dbc` parses; `Database::from_dbc` resolves value tables,
  comments, `GenMsgCycleTime`, `GenSigStartValue`, `VFrameFormat`,
  `SIG_VALTYPE_` floats and multiplexing (`M`/`mN` and `SG_MUL_VAL_` ranges).
  Bit layout: Intel signals occupy `start..start+size`; Motorola signals are
  `size` bits from the start bit's position in most-significant-first order.
  The tests cross-check payloads against `cantools`.
* `sbf`: own framing (`$@`, length a multiple of 4, CRC-16/XMODEM, resync one
  byte after a bad CRC) so every block keeps its raw bytes; bodies go to
  `libsbf::parse_datagram` and `sbf/fields.rs` maps common navigation blocks to
  fields. Encoders follow the SBF reference layouts (PVTGeodetic rev 2,
  AttEuler rev 1).

## Toward the scenario library

A scenario-side "HIL bridge" opens one channel per device: each simulation
step turns ground truth (CARLA ego pose and velocity) into device traffic
(`tcp://...?listen` + `sbf` for a GNSS receiver, `can://` + `dbc:` cyclic
messages via `every()`), and decodes the device under test's output back into
observations. Next: serial transport, E2E counters/CRCs in cyclic messages,
candump/BLF logs, more formats (NMEA, UBX, RTCM) behind the same traits.
