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

| Codec | |
| --- | --- |
| `raw` (default) | none: messages carry `data` (and `id` for CAN) |
| `dbc:FILE` | CAN frames <-> DBC messages, via [`can-dbc`](https://crates.io/crates/can-dbc) (options `strict`, `scaling`, `decode_choices`) |
| `sbf` | Septentrio Binary Format stream <-> blocks, via [`libsbf`](https://crates.io/crates/libsbf); encodes `PVTGeodetic` and `AttEuler` |

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
| `hilspy.open(url, codec="raw", **codec_options) -> Channel` | open a channel |
| `Channel.send(message)` / `send(name, fields)` | encode and send |
| `Channel.recv(timeout=None) -> Message \| None` | next message (iterate the channel for all) |
| `Channel.every(period, message_or_name, fields=None, *, duration=None) -> Periodic` | cyclic send from a Rust thread; `update()`, `stop()`, `period`, `sent_count` |
| `Channel.close()` | also via `with` / `async with` |
| `await Channel.asend(...)`, `await Channel.arecv(timeout=None)`, `async for msg in channel` | asyncio |
| `hilspy.merge(*channels)` | `(channel, message)` from several buses as they arrive (`for` / `async for`) |
| `hilspy.Message(name=None, fields=None, *, data=None, id=None, extended=None, fd=False)` | `.name`, `.fields`, `msg["field"]`, `.data`, `.id`, `.timestamp` |
| `hilspy.register_transport(scheme, factory)` / `register_codec(name, factory)` | plugins |
| `hilspy.Error` | bad URL/codec/option, encode/decode failure, closed channel |

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
