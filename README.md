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
| `mem://name` | in-process bus (tests, simulated ECUs in one process) |
| `file://path` | replay a recording; `?write` / `?append` records one |
| `serial:///dev/ttyUSB0` | serial port (`?baud=115200`, `data_bits`, `parity`, `stop_bits`, `flow`); `serial://COM3` on Windows |

| Codec | |
| --- | --- |
| `raw` (default) | none: messages carry `data` (and `id` for CAN) |
| `dbc:FILE` | CAN frames <-> DBC messages, via [`can-dbc`](https://crates.io/crates/can-dbc) (options `strict`, `scaling`, `decode_choices`) |
| `sbf` | Septentrio Binary Format stream <-> blocks, via [`libsbf`](https://crates.io/crates/libsbf); encodes `PVTGeodetic` and `AttEuler` |
| `nmea` | NMEA 0183 sentences <-> messages via [`nmea`](https://crates.io/crates/nmea) (`GGA`, `RMC`, `GSV`, ...); encodes `GGA`, `RMC`, `VTG`, `HDT`, `ZDA` |
| `ubx` | u-blox UBX frames <-> messages via [`ublox`](https://crates.io/crates/ublox) (`NAV-PVT`, `NAV-ATT`, `MON-VER`, ...); encodes `NAV-PVT` |
| `rtcm` | RTCM 3 corrections <-> messages via [`rtcm-rs`](https://crates.io/crates/rtcm-rs) (`1005`, `1077`, ...), both ways for every message it knows |
| `lidar:MODEL` | LiDAR UDP data packets <-> point clouds: Hesai and Velodyne line-ups, described by [profiles](#lidar-hesai-velodyne) |

## Install

Python 3.10+. One abi3 wheel per platform serves every Python version; wheels
are built for Linux (x86_64, aarch64), macOS (arm64, x86_64) and Windows x64.
`can://` (SocketCAN) is Linux-only; every other transport and codec works everywhere.

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
| `Channel.every(period, message_or_name, fields=None, *, duration=None) -> Periodic` | cyclic send from a Rust thread; `update()`, `stop()`, `period`, `sent_count` |
| `Channel.close()` | also via `with` / `async with` |
| `await Channel.asend(...)`, `await Channel.arecv(timeout=None)`, `async for msg in channel` | asyncio |
| `hilspy.merge(*channels)` | `(channel, message)` from several buses as they arrive (`for` / `async for`) |
| `hilspy.Message(name=None, fields=None, *, data=None, id=None, extended=None, fd=False)` | `.name`, `.fields`, `msg["field"]`, `.data`, `.id`, `.timestamp` |
| `hilspy.Clock(start=0.0)` | simulated time: `advance(dt)`, `advance_to(t)`, `now`, `next_due` |
| `hilspy.register_transport(scheme, factory)` / `register_codec(name, factory)` | plugins |
| `hilspy.Error` | bad URL/codec/option, encode/decode failure, closed channel |

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
```

## Development

```sh
uv sync                                 # build + dev dependencies (cantools for cross-checks)
cargo test -p hilspy-core               # Rust unit tests
uv run pytest                           # Python tests
HILSPY_TEST_VCAN=vcan0 uv run pytest tests/test_socketcan.py   # needs a vcan interface
uv run pre-commit run --all-files       # ruff, mypy, cargo fmt, clippy
```

Design notes: [`docs/design.md`](docs/design.md).

## License

Apache-2.0
