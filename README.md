# hilspy

HILS (hardware-in-the-loop simulation) I/O for Python on a Rust core, built to
be embedded in a scenario library (such as
[`autoware_carla_scenario`](https://github.com/hakuturu583/autoware_carla_scenario)).

One call opens a **channel**: a **transport** (how packets travel) composed
with a **codec** (what they mean). Both are pluggable, in Rust or in Python.

```python
import hilspy

with hilspy.open("can://vcan0", "dbc:vehicle.dbc") as ecu:
    ecu.send("VehicleSpeed", {"Speed": 12.5, "Accel": 0.0, "Gear": "D", "Counter": 0})
    for msg in ecu:  # until ecu.close()
        print(msg.name, msg.fields)  # "SteeringCmd" {"SteeringAngle": -3.5, ...}
```

| Transport URL | |
| --- | --- |
| `can://vcan0` | Linux SocketCAN (`?fd`, `?echo`, `?filter=0x100:0x700,...`) |
| `udp://host:port` | UDP to a peer (`?bind=addr:port`); `udp://addr:port?listen` replies to the last sender |
| `tcp://host:port` | TCP client; `tcp://addr:port?listen` serves every connected client |
| `mem://name` | in-process bus (tests, simulated ECUs in one process); carries CAN, LIN and FlexRay frames with their headers |
| `file://path` | replay a recording; `?write` / `?append` records one |
| `file://drive.blf`, `drive.asc`, `drive.log?format=candump` | CAN logs (Vector BLF/ASC, candump) with identifiers and times; `?realtime` replays them as recorded, `?write` records |
| `serial:///dev/ttyUSB0` | serial port (`?baud=115200`, `data_bits`, `parity`, `stop_bits`, `flow`); `serial://COM3` on Windows |
| `v4l2:///dev/video10` | camera frames into a V4L2 device (v4l2loopback) through GStreamer (`?width=`, `height`, `format`, `fps`, `output`); Linux, built with the `v4l2` feature, see [Cameras](#cameras-v4l2loopback) |

| Codec | |
| --- | --- |
| `raw` (default) | none: messages carry `data` (and `id` for CAN) |
| `dbc:FILE` | CAN frames <-> DBC messages, via [`can-dbc`](https://crates.io/crates/can-dbc) (options `strict`, `scaling`, `decode_choices`, `e2e`, `ecu`) |
| `fibex:FILE` | CAN / CAN FD, LIN and FlexRay frames <-> the frames of an ASAM FIBEX channel, via [`afibex`](https://crates.io/crates/afibex) (options `channel`, `schedule`, and the `dbc:` options); see [FIBEX](#fibex-networks) |
| `someip:FILE` | SOME/IP messages <-> the methods, events and fields of ASAM FIBEX 4 services, parsed by [`afibex`](https://crates.io/crates/afibex) and checked against [`asomeip`](https://crates.io/crates/asomeip) (options `client_id`, `decode_choices`, `stream`); see [SOME/IP](#someip-services) |
| `sbf` | Septentrio Binary Format stream <-> blocks, via [`libsbf`](https://crates.io/crates/libsbf); encodes `PVTGeodetic` and `AttEuler` |
| `nmea` | NMEA 0183 sentences <-> messages via [`nmea`](https://crates.io/crates/nmea) (`GGA`, `RMC`, `GSV`, ...); encodes `GGA`, `RMC`, `VTG`, `HDT`, `ZDA` |
| `ubx` | u-blox UBX frames <-> messages via [`ublox`](https://crates.io/crates/ublox) (`NAV-PVT`, `NAV-ATT`, `MON-VER`, ...); encodes `NAV-PVT` |
| `rtcm` | RTCM 3 corrections <-> messages via [`rtcm-rs`](https://crates.io/crates/rtcm-rs) (`1005`, `1077`, ...), both ways for every message it knows |
| `lidar:MODEL` | LiDAR UDP data packets <-> point clouds: Hesai and Velodyne line-ups, described by [profiles](#lidar-hesai-velodyne) |

## Install

Python 3.10+. One abi3 wheel per platform serves every Python version; wheels
are built for Linux (x86_64, aarch64), macOS (arm64, x86_64) and Windows x64.
`can://` (SocketCAN) is Linux-only; every other transport and codec works everywhere.
`v4l2://` (Linux) links the system GStreamer, so the published wheels leave it
out: build from source with the `v4l2` feature
([Cameras](#cameras-v4l2loopback) has a setup command that does it).

```sh
pip install hilspy
uv add hilspy
pip install git+https://github.com/hakuturu583/hilspy   # from source (needs Rust)
```

From a checkout:

```sh
uv sync                             # builds the extension with maturin
uv run pytest
```

Versions follow semver and are released by labelling the PR (`bump patch` /
`bump minor` / `bump major`); see [`docs/releasing.md`](docs/releasing.md).

## API

The whole public API:

| Name | |
| --- | --- |
| `hilspy.open(url, codec="raw", *, clock=None, **codec_options) -> Channel` | open a channel (on a virtual clock with `clock=`) |
| `Channel.send(message)` / `send(name, fields)` | encode and send |
| `Channel.recv(timeout=None) -> Message \| None` | next message (iterate the channel for all) |
| `Channel.every(period, message_or_name, fields=None, *, duration=None) -> Periodic` | cyclic send from a Rust thread; `update()`, `stop()`, `period`, `sent_count`. `period=None`: the message's cycle time from the DBC / FIBEX (see [Cycle times](#cycle-times)) |
| `Channel.close()` | also via `with` / `async with` |
| `await Channel.asend(...)`, `await Channel.arecv(timeout=None)`, `async for msg in channel` | asyncio |
| `hilspy.merge(*channels)` | `(channel, message)` from several buses as they arrive (`for` / `async for`) |
| `hilspy.Message(name=None, fields=None, *, data=None, id=None, bus=None, extended=None, fd=False, enhanced=None, checksum=None, cycle=None, channel=None)` | `.name`, `.fields`, `msg["field"]`, `.data`, `.id`, `.timestamp`; `data` is bytes, a uint8 numpy array (any shape, C order) or memoryview, or a sequence of ints; frame headers `.bus`, `.extended`, `.fd` (CAN), `.enhanced`, `.checksum` (LIN), `.cycle`, `.channel` (FlexRay); see [Frames](#can-lin-and-flexray-frames) |
| `hilspy.Clock(start=0.0)` | simulated time: `advance(dt)`, `advance_to(t)`, `now`, `next_due` |
| `hilspy.register_transport(scheme, factory)` / `register_codec(name, factory)` | plugins |
| `hilspy.Error` | bad URL/codec/option, encode/decode failure, closed channel |

### CAN, LIN and FlexRay frames

A raw message's packet may be a frame of one of three buses; its header is
part of the `Message` (a `fibex:` codec on a [LIN](#lin-channels) or
[FlexRay](#flexray-channels) channel fills and checks them):

| `bus` | `id` | Header |
| --- | --- | --- |
| `"can"` | CAN identifier | `extended` (29-bit), `fd` (CAN FD) |
| `"lin"` | frame identifier, 0..=63 (the protected identifier's parity bits are derived) | `enhanced` (checksum model: enhanced, else classic), `checksum` (the checksum byte) |
| `"flexray"` | slot, 1..=2047 | `cycle` (cycle counter, 0..=63), `channel` (`"A"`, `"B"` or `"AB"`) |

```python
hilspy.Message(data=b"\x01\x02", id=0x123)  # CAN (bus="can")
hilspy.Message(data=b"\x80\x01", id=0x10, bus="lin")  # enhanced checksum computed
hilspy.Message(data=b"\x80\x01", id=0x10, enhanced=False)  # classic
hilspy.Message(data=bytes(4), id=7, cycle=1, channel="A")  # FlexRay (bus="flexray")
```

`bus` is implied by the keywords (`enhanced` / `checksum` for LIN, `cycle` /
`channel` for FlexRay, else CAN when there is an `id`). A LIN frame's `data`
are its data bytes; the checksum is computed under its model unless given
(diagnostic frames `0x3C` / `0x3D` are always classic). A FlexRay payload is an
even number of bytes, up to 254. `mem://` carries every header as is (a bad
LIN checksum arrives bad); there is no LIN or FlexRay hardware transport yet.
`can://` and CAN logs refuse LIN and FlexRay frames, and the byte transports
(`udp://`, `tcp://`, `serial://`, raw `file://`) carry the bytes only.

### Cycle times

`every(None, ...)` sends a message at the period its network description
gives it: a DBC's `GenMsgCycleTime` (milliseconds) or, for `fibex:`, the
`REPEATING-TIME-RANGE` of the frame triggering's `CYCLIC-TIMING` (an
`xs:duration` such as `PT0.01S`, so per channel); on a LIN channel, its
schedule table (see [LIN channels](#lin-channels)); on a FlexRay channel, the
cycle length times the cycle repetition (see [FlexRay channels](#flexray-channels)). A message without one (or
with a cycle time of 0) raises `hilspy.Error` naming it; other codecs have
no cycle times. The `Periodic`'s `period` can still be changed afterwards.

```python
with hilspy.open("can://can0", "dbc:vehicle.dbc") as ecu:
    speed = ecu.every(None, "VehicleSpeed", {"Speed": 0.0, "Gear": "D"})  # every 10 ms
```

### Acting as an ECU

`ecu="Name"` (for `dbc:` and `fibex:`) makes the channel one ECU of the
network:

```python
with hilspy.open("can://can0", "dbc:vehicle.dbc", ecu="VCU") as vcu:
    vcu.send("VehicleSpeed", {"Speed": 12.5, "Gear": "D"})  # the VCU transmits it
    vcu.send("SteeringCmd", {"Enable": 1})  # hilspy.Error: SteeringCmd is sent by ADAS, not VCU
    for msg in vcu:  # decoded: what the VCU receives; nameless: the rest
        ...
```

* Sending: only messages the ECU transmits encode; any other is an encode
  error naming its transmitter(s).
* Receiving: only messages the ECU receives decode; the others arrive as
  nameless messages with their identifier and bytes, like unknown frames.
* A DBC's transmitters are the `BO_` sender and any `BO_TX_BU_` ones, and its
  receivers those of the message's signals (`Vector__XXX` is none). A FIBEX
  frame's are the ECUs whose `CONNECTORS` have an `OUTPUT-PORT` /
  `INPUT-PORT` with a `FRAME-TRIGGERING-REF` to the channel's triggering of
  it. ECUs are named by `SHORT-NAME` (`BU_` for a DBC); an unknown name is an
  error listing the network's ECUs.
* Decoded messages carry no transmitter field.

### FIBEX networks

`fibex:FILE` sends and receives the CAN, CAN FD, LIN and FlexRay frames of one channel of an
ASAM FIBEX (MCD-2 NET) file, like `dbc:` does a DBC's:

```python
with hilspy.open("can://can0", "fibex:network.xml", channel="Chassis") as ecu:
    ecu.send("VehicleStatus", {"Speed": 12.5, "Gear": "D", "BrakePressure": 3.0, "BrakeActive": 0})
    wheels = {"Front.Left": 10.0, "Front.Right": 10.0, "Rear.Left": 10.0, "Rear.Right": 10.0}
    cyclic = ecu.every(0.01, "WheelSpeeds", wheels)  # fields of two PDUs, qualified
```

* `channel` selects a channel by short name or ID; it is optional when the
  file has one channel (otherwise the error lists them).
* Messages are named by frame. Identifiers come from the channel's frame
  triggerings (`EXTENDED-ADDRESSING="true"` or an identifier over `0x7FF`
  for extended frames, `CAN-FRAME-TX-BEHAVIOR`/`RX-BEHAVIOR` `CAN-FD` for
  CAN FD).
* Fields are the signals of all the frame's PDUs; a name two PDUs share
  becomes `PDU.Signal`. A signal's `BIT-POSITION` is read as a DBC start bit
  within its PDU: the least significant bit for Intel, the most significant
  for Motorola (`IS-HIGH-LOW-BYTE-ORDER`). A PDU instance's `BIT-POSITION`
  is the PDU's least significant bit in the frame, at any bit; a high-low
  instance holds the PDU big-endian, its `BIT-POSITION` the PDU's most
  significant bit as for a Motorola signal (so the PDU's last byte comes
  first, and its Intel signals are Motorola in the frame and vice versa).
  Off a byte boundary, a Motorola signal that crosses a byte of its PDU is
  split in the frame and rejected, as is a signal past the end of the frame.
* Multiplexed PDUs (`MULTIPLEXER`) work like DBC multiplexing: the `SWITCH`
  is the multiplexor field, the signals of each switched PDU of the
  `DYNAMIC-PART` are present for its `SWITCH-CODE`s, and the `STATIC-PART`'s
  signals always. Switched and static PDUs use the multiplexed PDU's bit
  positions (as AUTOSAR copies their segments, and `canmatrix` writes them).
* Coded types: unsigned, two's complement (`2C`, `SIGNED`), sign-magnitude
  (`SM`: the top bit is the sign) and one's complement (`1C`) integers of up
  to 64 bits, and IEEE floats; a negative zero decodes to 0.
* Codings: `IDENTICAL`, `LINEAR`, `SCALE-LINEAR`, `TEXTTABLE`, `RAT-FUNC`
  and `SCALE-RAT-FUNC` compu methods; physical constraints are the range
  `strict` checks. A texttable range decodes any raw value in it to its name
  and encodes the name to its lower limit. Piecewise scales decode by the
  scale whose raw interval holds the value and encode by inverting the first
  scale that reaches the physical value; a rational function encodes where it
  is linear or strictly monotonic over its scale (otherwise sending the signal
  fails, naming it). A single linear function with exact texttable entries is
  a plain DBC factor, offset and value table.
* `strict`, `scaling`, `decode_choices`, `e2e` and `ecu` work as for `dbc:`
  (see [Acting as an ECU](#acting-as-an-ecu) for FIBEX connectors); frames
  the channel lacks decode as nameless messages. `FORMULA` compu methods,
  nested multiplexing, switched or static PDU instances off bit 0 or
  high-low, and encodings other than those above are rejected when the file
  is opened.

#### LIN channels

A channel of a cluster whose `PROTOCOL` is `LIN` sends and receives LIN
frames (on `mem://` or a plugin transport for now: hilspy has no LIN hardware
transport yet):

```python
master = hilspy.open("mem://body-lin", "fibex:network.xml", channel="Lin", ecu="Bcm", clock=clock)
lamp = hilspy.open("mem://body-lin", "fibex:network.xml", channel="Lin", ecu="Lamp")
cmd = master.every(None, "LightCmd", {"Brightness": 50.0, "Mode": "High"})  # schedule table period
msg = lamp.recv()  # msg.bus == "lin", msg.id == 0x10, msg.checksum == 0x49
```

* The frame triggering's `IDENTIFIER-VALUE` is the frame identifier
  (0..=63); signals are laid out as on CAN.
* Checksums: the triggering's `CHECKSUM-TYPE` (or `LIN-CHECKSUM`) `CLASSIC` /
  `ENHANCED`, else classic for a cluster `PROTOCOL-VERSION` 1.x and enhanced
  otherwise; the diagnostic frames `0x3C` / `0x3D` are always classic.
  Sending computes the checksum; receiving verifies it under the frame's model
  (not the model the sender claims). A mismatch raises `hilspy.Error` from
  `recv` (or reaches a subscription as an error) naming the frame and both
  checksums, and the next frame decodes as usual: a frame failing its
  checksum has no trustworthy payload, as a CAN frame failing its CRC never
  arrives, whereas E2E checks an intact payload and reports in the `e2e` field.
* Cycle times come from the `SCHEDULE-TABLE`s of the channel (or its
  cluster): an entry's `DELAY` (`xs:duration`) is the time from its start to
  the next entry's, the table's period the sum of its delays, and a frame's
  cycle time the period divided by the number of entries that send it (exact
  for evenly spaced entries, their mean interval otherwise). The table is the
  one the `schedule` option names (short name or ID), else the first table
  that sends the frame; a frame no table sends falls back to its
  `CYCLIC-TIMING`. `every(None, ...)` therefore runs each frame at its
  schedule rate; it does not run the table's slots in order.
* Not covered: the master/slave header-response split (a sender sends the
  whole frame), event-triggered and sporadic frame collision resolution, and
  LDF files.
* Not yet checked against a tool-exported FIBEX file: the schedule table and
  checksum element names above follow the test fixture
  (`tests/data/lin_flexray.xml`), not a published schema, and some tools
  describe LIN schedules as `RELATIVELY-SCHEDULED-TIMING`s instead. A file
  laid out differently still decodes, but may lack cycle times or fall back
  to the protocol version's checksum model.

#### FlexRay channels

A channel of a cluster whose `PROTOCOL` is `FlexRay` sends and receives
FlexRay frames (again on `mem://` or a plugin transport: there is no FlexRay
hardware transport):

```python
clock = hilspy.Clock()
esc = hilspy.open("mem://chassis-fr", "fibex:network.xml", channel="FlexRayA", clock=clock)
yaw = esc.every(None, "YawFr", {"YawRate": 1.5, "Valid": 1})  # 4 cycles of 5 ms
clock.advance(0.04)  # YawFr in cycles 1, 5 and 9: msg.id == 7 (slot), msg.cycle, msg.channel == "A"
```

* A frame triggering's `ABSOLUTELY-SCHEDULED-TIMING`s give its `SLOT-ID`,
  `BASE-CYCLE` (default 0) and `CYCLE-REPETITION` (a power of two up to 64,
  or 5, 10, 20, 40 or 50 for FlexRay 3.0; default 1): the frame is sent in that slot of every cycle whose counter
  modulo the repetition is the base cycle. Frames may share a slot in
  different cycles (slot multiplexing).
* The channel's `FLEXRAY-CHANNEL-NAME` (`A` or `B`; both when absent) is the
  frames' channel. Receiving decodes a frame seen on that channel (`AB`
  matches both) whose slot and cycle counter match one of the channel's
  frames; others (another slot, or the right slot in a cycle the frame is not
  sent in) arrive nameless.
* Sending pads the payload to whole two-byte words and picks the cycle
  counter from the message's time (the channel's clock, else the wall clock):
  the first cycle at or after `floor(time / cycle length) % 64` in which the
  frame is sent.
* The cycle length is the cluster's `CYCLE` (microseconds), else
  `MACRO-PER-CYCLE` times `MACROTICK`; a frame's cycle time is the cycle
  length times its (first timing's) repetition.
* Static and dynamic slots are treated alike: a dynamic-segment frame is
  decoded and encoded by its slot, without minislot arbitration. Startup,
  sync frames, null frames and the header CRC are not modelled.

### SOME/IP services

`someip:FILE` sends and receives the SOME/IP messages of the services an ASAM
FIBEX 4 file describes, over `udp://` or `tcp://`:

```python
origin = {"latitude": 35.68, "longitude": 139.69, "altitude": 40.0, "quality": "RTK_FIXED"}
route = {"origin": origin, "waypoints": [{"id": 1, "speed": 20}], "name": "Route 1"}
with hilspy.open("udp://192.168.0.20:30490", "someip:services.xml", client_id=0x42) as ecu:
    ecu.send("VehicleService.SetRoute", route)
    for msg in ecu:  # responses, events and field notifications
        print(msg.name, msg["message_type"], msg["session_id"], msg.fields)
```

* Messages are named `Service.Method` and `Service.Event` (FIBEX short
  names); a field's accessors are `Service.get_F_field`, `Service.set_F_field`
  and its notifier `Service.changed_F_field`.
* Fields: the header (`service_id`, `method_id`, `length`, `client_id`,
  `session_id`, `protocol_version`, `interface_version`, `message_type`,
  `return_code`) and the parameters: input parameters for requests and
  notifications, return parameters for responses (`"message_type":
  "RESPONSE"`). Structures and unions are dicts, arrays lists or
  `array.array`s, enums their names (`decode_choices=False` for numbers).
* Sending fills the header: `message_type` `REQUEST` (methods, accessors) or
  `NOTIFICATION` (events, notifiers), `return_code` `E_OK`, `client_id` from
  the option (default 0; notifications 0, as PRS_SOMEIP_00925 requires),
  `session_id` counting per method from 1. Pass any
  of them to override (a response echoes the request's `client_id` and
  `session_id`). A received message sends back as the same bytes.
* Serialization follows the FIBEX attributes: byte order, bit lengths,
  length fields of strings, structures, unions and dynamic arrays
  (`LENGTH-FIELD-SIZE`, `ARRAY-LENGTH-FIELD-SIZE`, `TYPE-FIELD-SIZE`), fixed
  arrays, UTF-8/UTF-16 and fixed-length strings.
* A datagram may carry several messages; a TCP stream is cut by the length
  field (a datagram's incomplete remainder is an error, never joined to the
  next one; `stream=True` treats source-less packets such as a `file://`
  replay as a stream). Unsigned 64-bit values above 2^63 are Python ints
  too. Unknown services or methods, SOME/IP-TP segments and SOME/IP-SD
  (service discovery) arrive as nameless messages with their bytes.

### AUTOSAR E2E (alive counter + CRC)

The `dbc` codec protects chosen messages with an AUTOSAR E2E profile
([`autosar-e2e`](https://crates.io/crates/autosar-e2e): profiles 4, 5, 6, 7,
8, 11, 22). Sending fills the counter (counting per message) and CRC;
receiving checks them and adds an `e2e` field (`Ok`, `CrcError`,
`Repeated`, `WrongSequence`, ...):

```python
e2e = {
    "BrakeCmd": {"profile": 11, "data_id": 0x123, "signals": ["Crc", "Counter", "DataIdNibble"]},
    "SafetyStatus": {"profile": 5, "data_id": 0x4242, "offset": 0, "signals": ["Crc", "Counter"]},
}
with hilspy.open("can://can0", "dbc:chassis.dbc", e2e=e2e) as ecu:
    brake = ecu.every(0.01, "BrakeCmd", {"Pressure": 12.5, "Request": 1})  # counter + CRC per cycle
    for msg in ecu:
        if "e2e" in msg and msg["e2e"] != "Ok":
            print("E2E", msg.name, msg["e2e"])
```

`signals` names the signals the protection writes, so they may be left out
even with `strict`. Offsets are AUTOSAR's (bits); profile 11 takes `mode`
(`nibble`/`both`), `crc_offset`, `counter_offset`, `nibble_offset`; 22
`data_id_list` (16 values); all take `max_delta_counter`.

### Synchronous simulation (virtual clock)

Devices that do I/O on a fixed period (cyclic CAN frames, a GNSS receiver at
10 Hz, a LiDAR streaming packets through each revolution) can run on
simulated time instead of the wall clock. Open their channels with a shared
`hilspy.Clock`; every send then happens inside `clock.advance(dt)`, in time
order, when the simulation steps past it:

```python
clock = hilspy.Clock(start=carla_world_time)
ecu = hilspy.open("can://vcan0", "dbc:vehicle.dbc", clock=clock)
lidar = hilspy.open("udp://192.168.1.100:2368", "lidar:pandar_xt32", clock=clock)
speed = ecu.every(0.01, "VehicleSpeed", {"Speed": 0.0, "Gear": "D"})
scans = lidar.every(0.1, "scan", first_scan)

while running:
    world.tick()  # e.g. CARLA synchronous mode, 50 ms
    speed.update("VehicleSpeed", {"Speed": ego_speed(), "Gear": "D"})
    scans.update("scan", raycast_points())
    clock.advance(0.05)  # 5 speed frames, half a revolution of packets
```

On a clock:

- `every()` sends at once, then each period as the clock passes it (no
  thread, nothing between steps).
- A paced message's packets (a LiDAR revolution) go out as the clock reaches
  their times, so a 50 ms step sends the packets of 50 ms.
- Messages are stamped with the clock's time. Codecs write that time (the
  SBF TOW, a LiDAR scan's timestamps), and received messages carry the time
  they arrived at.

### asyncio and several buses at once

Every channel also has `await ch.asend(...)`, `await ch.arecv(timeout)`,
`async for msg in ch` and `async with`. `hilspy.merge(*channels)` receives from
many buses at once, yielding `(channel, message)` as messages arrive, with
`for` or `async for`:

```python
async with (
    hilspy.open("can://can0", "dbc:powertrain.dbc") as can0,
    hilspy.open("can://can1", "dbc:chassis.dbc") as can1,
    hilspy.open("tcp://192.168.3.1:28784", "sbf") as gnss,
    hilspy.merge(can0, can1, gnss) as buses,
):
    async for bus, msg in buses:
        if bus is gnss and msg.name == "PVTGeodetic":
            await can1.asend("GnssPosition", {"Lat": msg["latitude"], "Lon": msg["longitude"]})
```

Each bus is received on its own Rust thread, so awaiting never blocks the
event loop and a quiet or slow bus does not delay the others. A channel has
one consumer at a time: once received with `arecv`/`async for`/`merge`, its
synchronous `recv` refuses.

A message with a name goes through the codec; a nameless one
(`Message(data=b"...", id=0x123)`) is sent as is. Messages the codec does not
know (a CAN id missing from the DBC) arrive nameless.

### GNSS receiver simulation (SBF over TCP)

```python
with hilspy.open("tcp://0.0.0.0:28784?listen", "sbf") as receiver:
    receiver.send(
        "PVTGeodetic", {"mode": 4, "latitude": lat_rad, "longitude": lon_rad, "height": 40.0}
    )
    receiver.send("AttEuler", {"heading": 90.0, "pitch": 0.0, "roll": 0.0})
    # tow/wnc default to the current GPS time
```

### GNSS receivers (NMEA, UBX, RTCM) over serial

```python
clock = hilspy.Clock(start=time.time())
gps = hilspy.open("serial:///dev/ttyUSB0?baud=115200", "nmea", talker="GN", clock=clock)
fix = gps.every(
    0.1, "GGA", {"latitude": lat, "longitude": lon, "fix_type": "Rtk", "fix_satellites": 14}
)

ublox = hilspy.open("serial:///dev/ttyACM0", "ubx")
for msg in ublox:  # interleaved NMEA is skipped
    if msg.name == "NAV-PVT":
        print(msg["lat"], msg["lon"], msg["fix_type"])
```

Decoded fields are the parsing crate's (names and units, e.g. NAV-PVT in
degrees, metres and m/s); encoding takes the same fields, so a received
sentence or packet can be sent on, edited or not. Times default to the
message's time (the channel's clock). Repeated records (GSV satellites, MSM
signal data) arrive as lists of dicts.

### LiDAR (Hesai, Velodyne)

`lidar:MODEL` decodes a sensor's UDP data packets into point clouds, and
encodes point clouds into the packets a sensor would send (to stand in for
one in the loop). Sensors are **profiles**, not code:

| Vendor | Models (`lidar:<name>`; aliases in brackets) |
| --- | --- |
| Hesai | `pandar_40p`, `pandar_64`, `pandar_qt64`, `pandar_qt128`, `pandar_xt16`, `pandar_xt32`, `pandar_xt32m`, `pandar_128e3x` (Pandar128), `pandar_128e4x` (OT128), `pandar_at128` |
| Velodyne | `vlp16` (Puck), `vlp16_hires`, `vlp32c`, `vlp32mr`, `hdl32e`, `hdl64e`, `vls128` (Alpha Prime) |

Names ignore case, `-` and `_` (`lidar:VLP-16`, `lidar:XT32`).

```python
# Driver side: one message per revolution.
with hilspy.open("udp://0.0.0.0:2368?listen", "lidar:pandar_xt32") as lidar:
    for scan in lidar:
        xyz = np.column_stack([np.frombuffer(scan[k], np.float32) for k in "xyz"])

# Sensor side: send one revolution of a simulated point cloud.
with hilspy.open("udp://192.168.1.100:2368", "lidar:vlp16") as sensor:
    sensor.send("scan", {"x": xs, "y": ys, "z": zs, "intensity": 100})
```

A `scan` has arrays (`array.array`, readable by numpy without copying) `x`,
`y`, `z` (metres; x forward, y left, z up), `intensity`, `channel`,
`azimuth` (degrees, clockwise from forward), `distance`, `return` and `time`
(seconds after `timestamp`), plus `timestamp`, `model`, `return_mode`, `rpm`
and `packets`. Sending takes `x`/`y`/`z` (each point goes to the channel of
the nearest elevation) or `distance`/`azimuth`/`channel`, as lists, tuples,
`array.array` or numpy arrays; a received scan can be sent back as is.
Sending one revolution spreads its packets over the revolution's duration
(100 ms at 600 rpm), as the sensor does, so a driver's socket buffer is not
flooded; `send` returns when the last packet is out (`pace=False` sends
them at once). Codec options: `frame="packet"` (a message per packet),
`phase` (where scans start, degrees), `rpm`, `return_mode` and `pace`
(encoding), `profiles=FILE`.

Profiles are protobuf messages (schema:
[`lidar_profile.proto`](crates/hilspy-core/lidar/lidar_profile.proto)):
packet layout, per-channel elevation / azimuth offset / firing time, return
modes, timing and range. The defaults are written as text in
[`lidar/profiles/`](crates/hilspy-core/lidar/profiles) and committed compiled
as `lidar/profiles.binpb`, which hilspy embeds. To describe another sensor,
write a `ProfileSet` in text format, compile it with any protobuf tool
(`protoc --encode=hilspy.lidar.ProfileSet lidar_profile.proto < my_lidars.txtpb > my_lidars.binpb`)
and pass it with `profiles="my_lidars.binpb"`; it is searched before the
defaults. Channel angles are
nominal values from [tier4/nebula](https://github.com/tier4/nebula)'s
calibration files; per-unit calibration is not applied.

### Cameras (v4l2loopback)

`v4l2://DEVICE` turns sent frames into a camera that any V4L2 client sees
(a browser, OpenCV, a ROS camera driver, `ffplay /dev/video10`): each packet
is one frame, pushed through `appsrc ! videoconvert ! videoscale ! v4l2sink`
([gstreamer-rs](https://crates.io/crates/gstreamer)) into a
[v4l2loopback](https://github.com/umlaeute/v4l2loopback) device.

```python
camera = hilspy.open("v4l2:///dev/video10?width=640&height=480&fps=30")
camera.send(hilspy.Message(data=rgb))  # an (480, 640, 3) uint8 numpy array
jpeg_camera = hilspy.open("v4l2:///dev/video11?format=jpeg")
jpeg_camera.send(hilspy.Message(data=open("frame.jpg", "rb").read()))
```

| Option | |
| --- | --- |
| `format` | what a packet holds: a GStreamer raw format (`RGB`, the default, `BGR`, `RGBA`, `BGRA`, `GRAY8`, `I420`, `NV12`, `YUY2`, ...), or `jpeg` / `png` |
| `width`, `height` | frame size; required for raw formats, a size to scale to for `jpeg` / `png` |
| `fps` | nominal rate in the caps (`30`, `30000/1001`); frames go out as sent, so the caller (or `every()`) paces them |
| `output` | the raw format the device receives (default `YUY2`) |

Raw frames are tightly packed (a numpy image as is) or in GStreamer's padded
layout; a frame of another size raises `hilspy.Error`, and so does the send
after a pipeline error (a device that refuses the format, an image that does
not decode). The channel receives nothing.

The feature links the system GStreamer, so it is not in the published
wheels. It works on amd64 and arm64 Linux alike (CI tests both).
`tools/setup_host.py` sets a Debian / Ubuntu host (x86_64 or aarch64) up with
[ansible-runner](https://ansible.readthedocs.io/projects/runner/) (playbook
`tools/ansible/v4l2loopback.yml`): GStreamer and v4l2loopback (DKMS) are
installed, the devices are created now and at every boot
(`/etc/modprobe.d/hilspy-v4l2loopback.conf`, labels `hilspy camera N`,
`exclusive_caps=1` so browsers list them), you join the `video` group and,
with `--build`, this checkout is rebuilt with the feature:

```sh
uv run --group setup tools/setup_host.py --build              # /dev/video10
uv run --group setup tools/setup_host.py --devices 2 --first 20 --label "sim camera"
uv run --group setup tools/setup_host.py --check              # show what would change
uv run --no-sync python examples/camera_simulator.py v4l2:///dev/video10
```

The module is built by DKMS against the running kernel's headers:
`linux-headers-$(uname -r)` on distribution kernels (Ubuntu and Debian on
amd64 and arm64, Ubuntu for Raspberry Pi, Raspberry Pi OS bookworm),
`nvidia-l4t-kernel-headers` on NVIDIA Jetson (L4T) and
`raspberrypi-kernel-headers` on older Raspberry Pi OS; headers already at
`/lib/modules/$(uname -r)/build` (a vendor BSP) are used as they are.
`sudo` does the system changes (its password is asked for when needed).
After `--build`, use `uv run --no-sync` (or rebuild with
`MATURIN_PEP517_ARGS="--features v4l2" uv sync --reinstall-package hilspy`):
a plain `uv sync` after a Rust change builds without the feature. Outside a
checkout: `MATURIN_PEP517_ARGS="--features v4l2" pip install hilspy --no-binary hilspy`.

### Metrics and dashboard (OpenTelemetry, Prometheus, Perses)

The Rust core records OpenTelemetry metrics for every channel and exports
them over OTLP/HTTP once `OTEL_EXPORTER_OTLP_ENDPOINT` (or
`OTEL_EXPORTER_OTLP_METRICS_ENDPOINT`) is set; without it nothing is
exported. No code changes: the standard variables configure it.

| Metric (Prometheus name) | What |
| --- | --- |
| `hilspy_packets_total`, `hilspy_bytes_total` | packets and payload bytes per `bus` (the URL without its query), `scheme`, `codec`, `direction` (`tx` / `rx`) |
| `hilspy_messages_total` | messages sent or decoded, per `message` (the name, or `0x1A0` for a nameless CAN frame): the send / receive rate of each |
| `hilspy_can_bits_total` | nominal CAN frame bits (no stuff bits); over the bit rate it is the bus load |
| `hilspy_errors_total` | failed sends and receives per `error` (`encode`, `decode`, `io`, ...) |
| `hilspy_periodic_lateness_seconds`, `hilspy_periodic_skipped_total` | how late `every()` sends start, deadlines skipped |
| `hilspy_transport_send_duration_seconds` | time a transport takes one packet (a full CAN queue, TCP back-pressure) |
| `process_cpu_utilization_ratio`, `process_memory_usage_bytes` | load of the process (the ECU it plays) |
| `system_cpu_utilization_ratio` (per core), `system_memory_utilization_ratio`, `system_cpu_load_average_1m` | load of the host |

`OTEL_SERVICE_NAME` (default `hilspy`) names the process (Prometheus `job`),
`OTEL_RESOURCE_ATTRIBUTES` adds attributes, `OTEL_METRIC_EXPORT_INTERVAL`
sets the export period (default here 5000 ms); the `instance` is
`host:pid`. Only `http/protobuf` is supported (gRPC and TLS are not built
in; put an OpenTelemetry Collector in between for those).

The dashboard is a container image for amd64 and arm64,
`ghcr.io/hakuturu583/hilspy-dashboard:<version>`: Prometheus with its OTLP
receiver on port 9090 and Perses with the hilspy dashboard (buses, message
rates, CAN bus load, periodic timing, CPU and memory) on port 8080. The setup
command pulls the image of your hilspy version:

```sh
uv run tools/setup_host.py --dashboard --no-v4l2      # docker pull <image>:<this version>
docker run -d --name hilspy-dashboard -p 9090:9090 -p 8080:8080 \
  -v hilspy-prometheus:/prometheus ghcr.io/hakuturu583/hilspy-dashboard:<version>
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:9090/api/v1/otlp OTEL_SERVICE_NAME=brake-ecu \
  python my_simulation.py
# http://localhost:8080/projects/hilspy/dashboards/hilspy
```

Arguments after the image go to Prometheus (`--storage.tsdb.retention.time=30d`).
`--dashboard-version` pulls another release (`latest`). The image is built
from `dashboard/` by `.github/workflows/dashboard.yml` (on every PR; pushed
on release).

### Plugins

A transport is any object with `send(packet)` and `recv(timeout)` (and
optionally `close()`); a codec has `encode(message)` and `decode(packet)`.
Packets are nameless `Message`s.

```python
class SerialTransport:
    def __init__(self, address: str, options: dict[str, str]):
        self.port = serial.Serial(address, int(options.get("baud", 115200)))

    def send(self, packet):
        self.port.write(packet.data)

    def recv(self, timeout):
        self.port.timeout = timeout
        data = self.port.read(self.port.in_waiting or 1)
        return hilspy.Message(data=data) if data else None


hilspy.register_transport("serial", SerialTransport)
gnss = hilspy.open("serial:///dev/ttyACM0?baud=921600", "sbf")
```

Rust crates register through `hilspy_core::transport::register` and
`hilspy_core::codec::register`.

A virtual CAN interface for local testing:

```sh
sudo modprobe vcan
sudo ip link add dev vcan0 type vcan
sudo ip link set vcan0 mtu 72        # CAN FD
sudo ip link set up vcan0
```

## Examples

```sh
uv run python examples/virtual_ecu.py                              # cyclic DBC message, in-process
uv run python examples/virtual_ecu.py --url can://vcan0
uv run python examples/gnss_simulator.py "tcp://127.0.0.1:28784?listen"
uv run python examples/gnss_simulator.py "file://sim.sbf?write" --seconds 5
uv run python examples/sbf_dump.py file://sim.sbf
uv run python examples/multi_bus.py                                # asyncio: CAN + GNSS at once
uv run --no-sync python examples/camera_simulator.py v4l2:///dev/video10   # see Cameras
```

## Development

```sh
uv sync                                 # build + dev dependencies (cantools for cross-checks)
cargo test -p hilspy-core               # Rust unit tests
uv run pytest                           # Python tests
HILSPY_TEST_VCAN=vcan0 uv run pytest tests/test_socketcan.py   # needs a vcan interface
cargo test -p hilspy-core --features v4l2                      # needs the GStreamer dev packages
HILSPY_TEST_V4L2=/dev/video10 uv run --no-sync pytest tests/test_v4l2.py  # after setup_host.py --build
uv run pre-commit run --all-files       # ruff, mypy, cargo fmt, clippy
```

Design notes: [`docs/design.md`](docs/design.md).

## License

Apache-2.0
