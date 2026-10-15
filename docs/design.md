# Design

## Layers

```
Python API     python/hilspy/                 open, Channel (sync + asyncio), merge, Message, Periodic, Error, register_*
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

## Asynchronous reception and several buses

`Channel::subscribe(deliver)` receives on a dedicated thread and hands every
message (or receive error, then one `Closed`) to `deliver` as it arrives.
The Python layer builds on it: `arecv` / `async for` deliver into an
`asyncio.Queue` through `loop.call_soon_threadsafe`, and `merge(*channels)`
subscribes every channel into one queue (asyncio or thread-safe), tagged with
its channel. Awaiting therefore never blocks the event loop and buses are
received concurrently. A subscription is its channel's only consumer
(`recv` refuses meanwhile); dropping it (or the `merge`) stops its thread.
Sends from asyncio (`asend`) run in a worker thread because a transport may
block (a full CAN queue, TCP back-pressure).

## Virtual clock

`Clock` (`src/clock.rs`) is a priority queue of work keyed by `(due ns,
sequence)`. `advance_to(t)` pops what is due up to `t` in order, moving `now`
to each due time before running it, so work can schedule more work within
the same step; a lock keeps steps from several threads apart. A channel opened
on a clock:

* schedules `every()` as self-rescheduling ticks (first at once) instead of
  a thread; stop, update and period changes apply at the next tick;
* sends a paced message's first packet at once and schedules the rest at
  their timestamp offsets (whole microseconds), so a revolution is spread
  over the steps that cover it; a failed deferred send is reported by the
  next send;
* stamps messages to encode with the clock's time (codecs use
  `Message::timestamp` for the times they write) and received packets with
  the time of reception.

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
* `serial://`: the `serialport` crate (built without libudev, so wheels need
  no system library); a reader and a cloned writer handle, reads in 100 ms
  slices so close is noticed; packets are stream chunks like TCP's.
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
* `nmea`, `ubx`, `rtcm`: stream framing per packet source (sentences to
  `\n`; UBX sync, length and Fletcher checksum; RTCM through
  `rtcm_rs::next_msg_frame`), then the `nmea`, `ublox` and `rtcm-rs` crates
  parse; their serde representation becomes the fields (`value::from_json`:
  numeric arrays become typed arrays, other arrays `Value::List`). RTCM
  encodes through the crate as well (`to_json` -> serde -> `MessageBuilder`);
  the crates do not write NMEA sentences or NAV-PVT, so those encoders are
  ours, cross-checked against pynmeagps and pyubx2 in the tests.
* `lidar:MODEL`: data-driven. A sensor is a **profile**
  (`crates/hilspy-core/lidar/lidar_profile.proto`): the packet layout
  (size, constant/verified bytes, blocks of `azimuth + units`, unit layout,
  banks per firing, firings per block, distance unit, offsets of the return
  mode, motor speed, timestamp and sequence), per-channel elevation /
  azimuth offset / firing offset, return modes, firing timing and range.
  Profiles are authored as protobuf text (`lidar/profiles/*.txtpb`, channel
  angles from tier4/nebula), compiled to the committed binary
  `lidar/profiles.binpb` (`cargo run -p hilspy-core --example
  lidar_profiles`; `tests/lidar_profiles.rs` fails when it is stale) and
  embedded with `include_bytes!`. `build.rs` generates the prost types with
  protox, so no `protoc` is needed. One generic engine (`src/lidar/`) reads
  every layout: Hesai Pandar40P/64/QT64/QT128/XT16/XT32/XT32M/128E3X/128E4X/
  AT128 and Velodyne VLP-16 (+Hi-Res)/VLP-32C/VLP-32MR/HDL-32E/HDL-64E/
  VLS-128. A new sensor is a new profile (`profiles=FILE` adds a
  `ProfileSet` at run time), not new code.
  Decoding yields per-point columns (`Value::Array`, `array.array` in
  Python) and cuts scans where the azimuth passes `phase` (or one message per
  packet). Encoding grids a point cloud into the firing columns of one
  revolution (nearest point per channel and column, whole packets per
  revolution) for simulating a sensor in the loop; `Codec::paced` makes the
  channel send them spaced by their timestamps, as the sensor streams them. Angles are nominal:
  per-unit calibration (distance corrections, AT128 mirror tables) is not
  modelled.

## Toward the scenario library

A scenario-side "HIL bridge" opens one channel per device: each simulation
step turns ground truth (CARLA ego pose and velocity) into device traffic
(`tcp://...?listen` + `sbf` for a GNSS receiver, `can://` + `dbc:` cyclic
messages via `every()`), and decodes the device under test's output back into
observations. Next: serial transport, E2E counters/CRCs in cyclic messages,
candump/BLF logs behind the same traits.
A simulated LiDAR is `udp://driver:2368` + `lidar:MODEL`, sending the
raycast point cloud of each revolution.
