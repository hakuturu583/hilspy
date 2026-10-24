use std::time::Duration;

use opentelemetry::metrics::MeterProvider;
use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData, ResourceMetrics};
use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};

use super::*;
use crate::channel::Channel;
use crate::value::{Options, Value};
use crate::{codec, transport};

/// A provider of its own per test: the global one is process-wide.
struct Recorder {
    provider: SdkMeterProvider,
    exporter: InMemoryMetricExporter,
}

impl Recorder {
    fn new() -> Self {
        let exporter = InMemoryMetricExporter::default();
        let provider = SdkMeterProvider::builder()
            .with_reader(PeriodicReader::builder(exporter.clone()).build())
            .build();
        Self { provider, exporter }
    }

    fn channel(&self, url: &str, codec: &str) -> Channel {
        let instruments = Arc::new(Instruments::new(&self.provider.meter(SCOPE)));
        let endpoint = crate::url::Endpoint::parse(url).unwrap();
        let name = codec.split_once(':').map_or(codec, |(n, _)| n);
        let mut options = Options::new();
        if codec.starts_with("dbc:") {
            options.insert("strict", Value::Bool(false));
        }
        Channel::build(
            transport::open(url).unwrap(),
            codec::open(codec, options).unwrap(),
            None,
            ChannelMetrics::new(instruments, &endpoint.scheme, &endpoint.address, name),
        )
    }

    fn collect(&self) -> Vec<ResourceMetrics> {
        self.provider.force_flush().unwrap();
        self.exporter.get_finished_metrics().unwrap()
    }

    /// The latest value of a sum, a gauge or a histogram's count whose
    /// attributes include `with`.
    fn value(&self, name: &str, with: &[(&str, &str)]) -> Option<f64> {
        let matches = |attributes: Vec<&KeyValue>| {
            with.iter().all(|(k, v)| {
                attributes
                    .iter()
                    .any(|kv| kv.key.as_str() == *k && kv.value.to_string() == *v)
            })
        };
        let mut found = None;
        for rm in self.collect() {
            for metric in rm.scope_metrics().flat_map(|s| s.metrics()) {
                if metric.name() != name {
                    continue;
                }
                let mut total = None::<f64>;
                let mut add = |v: f64| *total.get_or_insert(0.0) += v;
                match metric.data() {
                    AggregatedMetrics::U64(MetricData::Sum(s)) => s
                        .data_points()
                        .filter(|p| matches(p.attributes().collect()))
                        .for_each(|p| add(p.value() as f64)),
                    AggregatedMetrics::F64(MetricData::Histogram(h)) => h
                        .data_points()
                        .filter(|p| matches(p.attributes().collect()))
                        .for_each(|p| add(p.count() as f64)),
                    AggregatedMetrics::F64(MetricData::Gauge(g)) => g
                        .data_points()
                        .filter(|p| matches(p.attributes().collect()))
                        .for_each(|p| add(p.value())),
                    AggregatedMetrics::U64(MetricData::Gauge(g)) => g
                        .data_points()
                        .filter(|p| matches(p.attributes().collect()))
                        .for_each(|p| add(p.value() as f64)),
                    _ => {}
                }
                found = total.or(found);
            }
        }
        found
    }
}

fn dbc() -> String {
    format!(
        "dbc:{}/../../tests/data/vehicle.dbc",
        env!("CARGO_MANIFEST_DIR")
    )
}

#[test]
fn channels_count_what_they_send_and_receive() {
    let r = Recorder::new();
    let tx = r.channel("mem://t-otel-traffic?echo=0", &dbc());
    let rx = r.channel("mem://t-otel-traffic", &dbc());
    let speed = Message::new("VehicleSpeed", vec![("Speed".into(), Value::Float(12.5))]);
    for _ in 0..3 {
        tx.send(&speed).unwrap();
    }
    tx.send(&Message::raw(Packet::can(0x7FF, vec![1, 2])))
        .unwrap();
    for _ in 0..4 {
        rx.recv(Some(Duration::from_secs(1))).unwrap().unwrap();
    }
    // An unknown message fails to encode.
    assert!(tx.send(&Message::new("Nope", vec![])).is_err());

    let bus = ("bus", "mem://t-otel-traffic");
    let tx_ = ("direction", "tx");
    let rx_ = ("direction", "rx");
    assert_eq!(r.value("hilspy.packets", &[bus, tx_]), Some(4.0));
    assert_eq!(r.value("hilspy.packets", &[bus, rx_]), Some(4.0));
    assert_eq!(r.value("hilspy.bytes", &[bus, rx_]), Some(3.0 * 8.0 + 2.0));
    let speed = ("message", "VehicleSpeed");
    assert_eq!(r.value("hilspy.messages", &[bus, tx_, speed]), Some(3.0));
    assert_eq!(r.value("hilspy.messages", &[bus, rx_, speed]), Some(3.0));
    // The raw frame is unknown to the DBC: decoded nameless, by identifier.
    assert_eq!(
        r.value("hilspy.messages", &[rx_, ("message", "0x7FF")]),
        Some(1.0)
    );
    assert_eq!(
        r.value("hilspy.can.bits", &[rx_]),
        Some(3.0 * (47.0 + 64.0) + 47.0 + 16.0)
    );
    assert_eq!(
        r.value("hilspy.errors", &[tx_, ("error", "not_found")]),
        Some(1.0)
    );
    assert_eq!(r.value("hilspy.transport.send.duration", &[tx_]), Some(4.0));
    assert_eq!(
        r.value("hilspy.packets", &[("scheme", "mem"), ("codec", "dbc")]),
        Some(8.0)
    );
}

#[test]
fn periodic_sends_record_their_lateness() {
    let r = Recorder::new();
    let tx = r.channel("mem://t-otel-periodic", "raw");
    let frame = Message::raw(Packet::can(0x100, vec![0; 8]));
    let task = tx
        .every(Some(Duration::from_millis(5)), frame, None)
        .unwrap();
    std::thread::sleep(Duration::from_millis(100));
    task.stop();
    let late = r
        .value(
            "hilspy.periodic.lateness",
            &[("bus", "mem://t-otel-periodic"), ("message", "0x100")],
        )
        .unwrap();
    // Every send but the first (made by every() itself).
    assert_eq!(late, (task.sent_count() - 1) as f64);
    assert!(late >= 5.0);
}

#[test]
fn can_frame_bits() {
    let h = |extended, fd| CanHeader {
        id: 1,
        extended,
        fd,
        brs: false,
    };
    assert_eq!(can_bits(&h(false, false), 8), 111);
    assert_eq!(can_bits(&h(true, false), 8), 131);
    assert_eq!(can_bits(&h(false, true), 16), 56 + 128);
    assert_eq!(can_bits(&h(false, true), 64), 60 + 512);
    assert_eq!(can_bits(&h(true, true), 0), 75);
}

#[test]
fn error_kinds() {
    assert_eq!(Error::Closed.kind(), "closed");
    assert_eq!(Error::encode("x").kind(), "encode");
    assert_eq!(Error::Config(String::new()).kind(), "config");
}

#[cfg(feature = "otlp")]
#[test]
fn host_gauges() {
    let r = Recorder::new();
    register_host_metrics(&r.provider.meter(SCOPE));
    let memory = r.value("process.memory.usage", &[]).unwrap();
    assert!(memory > 0.0, "{memory}");
    let used = r.value("system.memory.utilization", &[]).unwrap();
    assert!((0.0..=1.0).contains(&used), "{used}");
    let cpu0 = r.value("system.cpu.utilization", &[("cpu.logical_number", "0")]);
    assert!(cpu0.is_some_and(|u| (0.0..=1.0).contains(&u)));
    assert!(r.value("process.cpu.utilization", &[]).is_some());
    assert!(r.value("system.cpu.load_average.1m", &[]).is_some());
}
