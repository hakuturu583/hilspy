//! Metrics through OpenTelemetry: what every channel sends and receives,
//! how punctual its periodic messages are, and how loaded the host (the ECU
//! a HILS process plays) is.
//!
//! The instruments come from the global meter provider
//! ([`opentelemetry::global`]), so without one they cost next to nothing.
//! With the `otlp` feature (default) the first channel opened calls
//! [`init_from_env`], which installs a provider exporting over OTLP/HTTP when
//! `OTEL_EXPORTER_OTLP_ENDPOINT` (or `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT`)
//! is set; a program can install its own provider before opening channels
//! instead. Prometheus receives OTLP itself (`--web.enable-otlp-receiver`):
//! `OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:9090/api/v1/otlp`.
//!
//! | Instrument | Unit | Attributes |
//! | --- | --- | --- |
//! | `hilspy.packets` | `{packet}` | `bus`, `scheme`, `codec`, `direction` (`tx`, `rx`) |
//! | `hilspy.bytes` | `By` | as `hilspy.packets` |
//! | `hilspy.messages` | `{message}` | as `hilspy.packets`, `message` (the name, or a nameless CAN frame's `0x1A0`) |
//! | `hilspy.errors` | `{error}` | as `hilspy.packets`, `error` ([`Error::kind`]) |
//! | `hilspy.can.bits` | `{bit}` | as `hilspy.packets`; nominal CAN frame bits, no stuff bits (bus load: rate / bit rate) |
//! | `hilspy.transport.send.duration` | `s` | as `hilspy.packets` (one packet into the transport) |
//! | `hilspy.periodic.lateness` | `s` | `bus`, `message`: how late a periodic send started (threads, not a virtual clock) |
//! | `hilspy.periodic.skipped` | `{deadline}` | `bus`, `message`: deadlines skipped after falling more than a period behind |
//!
//! `bus` is the transport URL without its query (`can://vcan0`). The host
//! gauges (`otlp`, [`register_host_metrics`]) follow the semantic
//! conventions: `process.cpu.utilization`, `process.memory.usage`,
//! `system.cpu.utilization` (per `cpu.logical_number`),
//! `system.memory.utilization` and `system.cpu.load_average.1m`.

#[cfg(feature = "otlp")]
mod host;
#[cfg(feature = "otlp")]
mod otlp;
#[cfg(test)]
mod tests;

use std::sync::{Arc, OnceLock};

use opentelemetry::metrics::{Counter, Histogram, Meter};
use opentelemetry::KeyValue;

use crate::error::Error;
use crate::message::{CanHeader, Message, Packet};

#[cfg(feature = "otlp")]
pub use host::register_host_metrics;
#[cfg(feature = "otlp")]
pub use otlp::{init_from_env, shutdown};

/// Without the `otlp` feature nothing is exported from here: the metrics
/// go to the provider the program installs. Returns `Ok(false)`.
#[cfg(not(feature = "otlp"))]
pub fn init_from_env() -> crate::error::Result<bool> {
    Ok(false)
}

/// Without the `otlp` feature there is nothing to flush.
#[cfg(not(feature = "otlp"))]
pub fn shutdown() {}

/// The instrumentation scope of every hilspy instrument.
pub const SCOPE: &str = "hilspy";

/// Upper bounds of the duration histograms, seconds: 10 µs to 1 s.
const DURATION_BOUNDS: [f64; 14] = [
    1e-5, 2.5e-5, 5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3, 2.5e-3, 5e-3, 1e-2, 2.5e-2, 5e-2, 0.1, 1.0,
];

pub(crate) struct Instruments {
    packets: Counter<u64>,
    bytes: Counter<u64>,
    messages: Counter<u64>,
    errors: Counter<u64>,
    can_bits: Counter<u64>,
    send_duration: Histogram<f64>,
    lateness: Histogram<f64>,
    skipped: Counter<u64>,
}

impl Instruments {
    pub(crate) fn new(meter: &Meter) -> Self {
        let duration = |name: &'static str, description: &'static str| {
            meter
                .f64_histogram(name)
                .with_unit("s")
                .with_description(description)
                .with_boundaries(DURATION_BOUNDS.to_vec())
                .build()
        };
        let counter = |name: &'static str, unit: &'static str, description: &'static str| {
            meter
                .u64_counter(name)
                .with_unit(unit)
                .with_description(description)
                .build()
        };
        Self {
            packets: counter("hilspy.packets", "{packet}", "Packets sent or received"),
            bytes: counter(
                "hilspy.bytes",
                "By",
                "Packet payload bytes sent or received",
            ),
            messages: counter("hilspy.messages", "{message}", "Messages sent or decoded"),
            errors: counter("hilspy.errors", "{error}", "Failed sends and receives"),
            can_bits: counter(
                "hilspy.can.bits",
                "{bit}",
                "Nominal bits of the CAN frames sent or received (no stuff bits)",
            ),
            send_duration: duration(
                "hilspy.transport.send.duration",
                "Time a transport took to take one packet",
            ),
            lateness: duration(
                "hilspy.periodic.lateness",
                "How late a periodic send started after its deadline",
            ),
            skipped: counter(
                "hilspy.periodic.skipped",
                "{deadline}",
                "Periodic deadlines skipped after falling more than a period behind",
            ),
        }
    }

    /// The instruments of the global meter provider, created after
    /// [`init_from_env`] so they belong to the provider it installs (an
    /// error there is reported by [`crate::Channel::open`]).
    pub(crate) fn global() -> Arc<Self> {
        static GLOBAL: OnceLock<Arc<Instruments>> = OnceLock::new();
        Arc::clone(GLOBAL.get_or_init(|| {
            let _ = init_from_env();
            Arc::new(Self::new(&opentelemetry::global::meter(SCOPE)))
        }))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Direction {
    Tx,
    Rx,
}

/// One channel's view of the instruments, its attributes made once
/// (`Arc<str>` values: cloning them for a message only counts references).
pub(crate) struct ChannelMetrics {
    instruments: Arc<Instruments>,
    /// `bus`, `scheme`, `codec`, `direction`, for `Tx` and `Rx`.
    attributes: [[KeyValue; 4]; 2],
}

impl ChannelMetrics {
    /// `bus` is `scheme://address` (the URL without its query, empty without
    /// a scheme); `codec` is the codec's name.
    pub(crate) fn new(
        instruments: Arc<Instruments>,
        scheme: &str,
        address: &str,
        codec: &str,
    ) -> Self {
        let bus: Arc<str> = if scheme.is_empty() {
            "".into()
        } else {
            format!("{scheme}://{address}").into()
        };
        let (scheme, codec): (Arc<str>, Arc<str>) = (scheme.into(), codec.into());
        let attributes = |direction: &'static str| {
            [
                KeyValue::new("bus", bus.clone()),
                KeyValue::new("scheme", scheme.clone()),
                KeyValue::new("codec", codec.clone()),
                KeyValue::new("direction", direction),
            ]
        };
        Self {
            attributes: [attributes("tx"), attributes("rx")],
            instruments,
        }
    }

    fn attributes(&self, direction: Direction) -> &[KeyValue; 4] {
        &self.attributes[direction as usize]
    }

    fn with(&self, direction: Direction, extra: KeyValue) -> [KeyValue; 5] {
        let [bus, scheme, codec, direction] = self.attributes(direction).clone();
        [bus, scheme, codec, direction, extra]
    }

    pub(crate) fn packet(&self, direction: Direction, packet: &Packet) {
        let i = &self.instruments;
        let a = self.attributes(direction);
        i.packets.add(1, a);
        i.bytes.add(packet.data.len() as u64, a);
        if let Some(h) = &packet.can {
            i.can_bits.add(can_bits(h, packet.data.len()), a);
        }
    }

    /// A message sent or decoded: by name, or a nameless CAN frame by its
    /// identifier; other nameless messages are counted as packets only.
    pub(crate) fn message(&self, direction: Direction, message: &Message) {
        let Some(label) = message_label(message) else {
            return;
        };
        self.instruments
            .messages
            .add(1, &self.with(direction, KeyValue::new("message", label)));
    }

    pub(crate) fn error(&self, direction: Direction, error: &Error) {
        if !matches!(error, Error::Closed) {
            self.instruments.errors.add(
                1,
                &self.with(direction, KeyValue::new("error", error.kind())),
            );
        }
    }

    pub(crate) fn send_duration(&self, seconds: f64) {
        self.instruments
            .send_duration
            .record(seconds, self.attributes(Direction::Tx));
    }

    pub(crate) fn periodic(&self, message: &Message) -> PeriodicMetrics {
        let name = message_label(message).unwrap_or_default();
        PeriodicMetrics {
            instruments: Arc::clone(&self.instruments),
            attributes: [
                self.attributes(Direction::Tx)[0].clone(),
                KeyValue::new("message", name),
            ],
        }
    }
}

/// The `message` attribute: the name, or a nameless CAN frame's identifier.
fn message_label(message: &Message) -> Option<String> {
    match (&message.name, message.packet.as_ref().and_then(|p| p.can)) {
        (Some(name), _) => Some(name.clone()),
        (None, Some(h)) => Some(format!("0x{:X}", h.id)),
        (None, None) => None,
    }
}

/// A periodic message's punctuality.
pub(crate) struct PeriodicMetrics {
    instruments: Arc<Instruments>,
    /// `bus`, `message`.
    attributes: [KeyValue; 2],
}

impl PeriodicMetrics {
    pub(crate) fn lateness(&self, seconds: f64) {
        self.instruments.lateness.record(seconds, &self.attributes);
    }

    pub(crate) fn skipped(&self, deadlines: u64) {
        self.instruments.skipped.add(deadlines, &self.attributes);
    }
}

/// Nominal bits of a CAN data frame with `len` data bytes: start of frame,
/// arbitration, control, data, CRC and its delimiter, acknowledgement, end of
/// frame and intermission, without stuff bits. A CAN FD frame is counted as
/// if at one bit rate (with BRS its data phase is faster).
pub(crate) fn can_bits(header: &CanHeader, len: usize) -> u64 {
    let data = 8 * len as u64;
    match (header.fd, header.extended) {
        // SOF, ID, RTR, IDE, r0, DLC = 19; CRC 15 + 1; ACK 2; EOF 7; IFS 3.
        (false, false) => 47 + data,
        // + SRR, 18 identifier bits, r1.
        (false, true) => 67 + data,
        // SOF, ID, RRS, IDE, FDF, res, BRS, ESI, DLC = 22 (41 extended);
        // stuff count 4; CRC 17 (up to 16 bytes) or 21; delimiters, ACK,
        // EOF, IFS 13.
        (true, extended) => {
            (if extended { 41 } else { 22 }) + 4 + if len > 16 { 21 } else { 17 } + 13 + data
        }
    }
}
