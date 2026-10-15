# hilspy

HILS (hardware-in-the-loop simulation) I/O for Python on a Rust core. It is
built to be embedded in a scenario library (such as
[`autoware_carla_scenario`](https://github.com/hakuturu583/autoware_carla_scenario)),
where a simulator has to talk to real or simulated ECUs and sensors.

The first release covers:

| Module | What it does | Rust crate used |
| --- | --- | --- |
| `hilspy.can` | CAN / CAN FD frames; buses on Linux SocketCAN (`can0`, `vcan0`) or an in-process virtual bus; receive filters; periodic (cyclic) transmission from a Rust thread | [`socketcan`](https://crates.io/crates/socketcan) |
| `hilspy.dbc` | DBC databases: encode physical signal values to payloads/frames and decode them back (scaling, value tables, signed/float signals, Intel and Motorola layouts, multiplexing, CAN FD messages) | [`can-dbc`](https://crates.io/crates/can-dbc) |
| `hilspy.sbf` | Septentrio Binary Format: a streaming decoder (resync, CRC check, typed fields for PVT/attitude/INS/covariance blocks) and encoders for PVTGeodetic/AttEuler so a simulator can stand in for a GNSS receiver | [`libsbf`](https://crates.io/crates/libsbf) |

The Python package is a thin layer: everything above runs in Rust
(`crates/hilspy-core`) and is exposed through PyO3 (`crates/hilspy-python`).
Blocking calls release the GIL; a blocking `recv` stays interruptible with
Ctrl-C.

## Install

Linux, Python 3.10+. One abi3 wheel serves every Python version.

```sh
uv add hilspy                       # or: pip install hilspy
```

From a checkout (needs a Rust toolchain):

```sh
uv sync                             # builds the extension with maturin
uv run pytest
```

## CAN

```python
from hilspy.can import Bus, Filter, Frame

with Bus("vcan0") as bus:                          # interface="socketcan" is the default
    bus.send(Frame(0x123, b"\x01\x02\x03"))
    frame = bus.recv(timeout=1.0)                  # Frame or None
    for frame in bus:                              # until bus.shutdown()
        ...
```

* `Bus(channel, "virtual")` joins an in-process bus: every `Bus` opened on the
  same channel name in the process receives the others' frames. No kernel
  interface or root needed, so it suits tests and simulated ECUs wired
  together in one process.
* `fd=True` opens a CAN FD socket; `Frame(id, data, is_fd=True, bitrate_switch=True)`.
* `filters=[Filter(0x100, 0x700), {"can_id": 0x18FF0001, "extended": True}]`
  (kernel filters on SocketCAN), `receive_own_messages=True`.
* `bus.send_periodic(frame, period, duration=None)` sends from a Rust thread
  with drift-free absolute deadlines; `task.modify(frame)` swaps the payload
  (counters, checksums), `task.period = 0.02`, `task.stop()`.

A virtual CAN interface for local testing:

```sh
sudo modprobe vcan
sudo ip link add dev vcan0 type vcan
sudo ip link set vcan0 mtu 72        # CAN FD
sudo ip link set up vcan0
```

## DBC

```python
from hilspy import dbc

db = dbc.load_file("vehicle.dbc")
frame = db.encode_frame("VehicleSpeed", {"Speed": 42.0, "Accel": 0.5, "Gear": "D", "Counter": 3})
bus.send(frame)

values = db.decode_frame(bus.recv())               # {"Speed": 42.0, "Gear": "D", ...}
db.decode("VehicleSpeed", payload, decode_choices=False, scaling=False)   # raw values
```

`encode` is strict by default: every signal must be given, values must lie in
the signal's `[min|max]`, and names must exist. `strict=False` fills missing
signals with `GenSigStartValue` (or 0). Raw values that do not fit in the
signal's bits always fail. `Message.cycle_time` reads `GenMsgCycleTime`, and
`Message.is_fd` reads `VFrameFormat` (or a payload over 8 bytes).

## SBF

```python
from hilspy import sbf

decoder = sbf.Decoder()
for chunk in serial_or_socket_chunks:
    for block in decoder.feed(chunk):
        if block.name == "PVTGeodetic":
            print(block["latitude"], block["longitude"], block["mode_name"])

blocks = sbf.read_file("capture.sbf")

# Simulated receiver output
wnc, tow = sbf.gps_time(time.time())
data = sbf.encode_pvt_geodetic(tow=tow, wnc=wnc, mode=4, latitude=lat_rad, longitude=lon_rad,
                               height=40.0, vn=1.0, ve=0.0, vu=0.0)
data += sbf.encode_att_euler(tow=tow, wnc=wnc, mode=4, heading=90.0, pitch=0.0, roll=0.0)
```

Every block keeps its `raw` bytes, `number`, `revision`, `tow` and `wnc`;
`fields` is a dict for the blocks with a field mapping (PVTGeodetic,
PVTCartesian, AttEuler, INSNavGeod, ReceiverTime, DOP, PosCovGeodetic,
VelCovGeodetic, EndOfPVT/Att/Meas) and `None` otherwise. Do-not-use values read
as `None`. `sbf.frame_block(number, revision, body)` frames any other block.

## Examples

```sh
uv run python examples/virtual_ecu.py                       # cyclic DBC message on a virtual bus
uv run python examples/virtual_ecu.py --interface socketcan --channel vcan0
uv run python examples/gnss_simulator.py --file sim.sbf --seconds 5
uv run python examples/gnss_simulator.py --tcp 28784        # serve like a receiver's TCP port
uv run python examples/sbf_dump.py sim.sbf
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
