# Design

## Layers

```
Python API          python/hilspy/{can,dbc,sbf}.py   re-exports, small helpers (gps_time, read_file)
                    python/hilspy/_hilspy.pyi        type stubs of the extension
PyO3 bindings       crates/hilspy-python             conversions, GIL release, exceptions
Rust core           crates/hilspy-core               all behaviour; no Python dependency
Parsers / I/O       can-dbc, libsbf, socketcan
```

The core is usable from Rust on its own, so a future Rust-side scenario
runner, or a real-time process, can reuse it without Python.

## CAN

* `Bus` wraps an `Arc<dyn BusBackend>`; every method takes `&self`, so one bus
  is shared by a receive loop and any number of periodic senders. Backends:
  * `socketcan` (Linux): a raw `CanSocket` or, with `fd=True`, a `CanFdSocket`.
    Receive waits with `poll(2)` in 100 ms slices so `shutdown()` wakes a
    receiver blocked without a timeout. Filters are kernel filters (the
    identifier kind is matched through the `CAN_EFF_FLAG` bit).
  * `virtual`: a process-wide registry of hubs by channel name; each endpoint
    has its own queue (bounded, oldest dropped), filters, and FD capability (a
    classic endpoint does not see FD frames, like a classic controller).
* `Frame` constructors validate identifier range and (FD) length, so an
  invalid frame never reaches a backend.
* `PeriodicTask` runs one thread per task with absolute deadlines (no drift);
  a deadline missed by more than a period is skipped, not burst. Send errors
  are recorded (`last_error`) without stopping the task. It is the place to
  add per-cycle payload updates (alive counters, CRCs) in Rust later.
* Python: `recv` releases the GIL in 50 ms slices and calls
  `PyErr_CheckSignals` between them, so Ctrl-C works during a blocking wait.

## DBC

`can-dbc` parses the file into an AST; `Database::from_dbc` resolves it into
messages and signals with value tables, comments, `GenMsgCycleTime`,
`GenSigStartValue`, `VFrameFormat`, `SIG_VALTYPE_` floats and multiplexing
(simple `M`/`mN`, and extended `SG_MUL_VAL_` ranges), rejecting signals that
do not fit in their message.

Bit layout (`dbc/bits.rs`): Intel signals occupy `start..start+size`; Motorola
signals are `size` consecutive bits from the start bit's position in
most-significant-first numbering. The test suite checks payloads against
`cantools` for random values of every signal kind.

Encoding uses exact integer arithmetic where the scaling is integral (64-bit
raw values exceed `f64`), rounds otherwise, and always rejects raw values that
do not fit. Decoding returns `int` when scaling is integral, `float` otherwise,
and value-table names when `decode_choices`.

## SBF

`sbf::Decoder` does its own framing so that every block, known or not, comes
out with its raw bytes, number, revision, TOW and WNc: find `$@`, check the
length (a multiple of 4, at least 8), wait for the block, check
CRC-16/XMODEM, and on a bad CRC resync one byte later. The body of a complete
block goes to `libsbf::parse_datagram`, and `sbf/fields.rs` maps the parsed
structs of common navigation blocks to name/value fields (do-not-use values
become `None`).

`sbf::encode` writes blocks in the layout of the SBF reference guide
(PVTGeodetic rev 2, AttEuler rev 1); the tests parse them back with `libsbf`.

## Toward the scenario library

The planned integration is a scenario-side "HIL bridge" that, each simulation
step, turns ground truth (ego pose and velocity from CARLA) into sensor and
ECU traffic (SBF over TCP/serial, cyclic CAN messages from a DBC) and decodes
the device under test's CAN output back into scenario observations. The
pieces this library adds for that, in order:

1. CAN (SocketCAN/vcan + virtual), DBC codec, SBF decode/encode -- this release.
2. Transports for SBF and other byte streams (TCP server/client, serial ports).
3. Rust-side cyclic messages with per-cycle callbacks (counters, E2E CRCs) and
   timestamped logging/replay (candump/BLF).
4. More formats behind the same pattern (e.g. NMEA, UBX, RTCM) via existing
   Rust crates.
