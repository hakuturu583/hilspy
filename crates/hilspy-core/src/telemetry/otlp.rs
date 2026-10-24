//! The OTLP/HTTP exporter, configured by the standard `OTEL_*` variables.

use std::env;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use opentelemetry::KeyValue;
use opentelemetry_otlp::{
    MetricExporter, OTEL_EXPORTER_OTLP_ENDPOINT, OTEL_EXPORTER_OTLP_METRICS_ENDPOINT,
    OTEL_EXPORTER_OTLP_METRICS_PROTOCOL, OTEL_EXPORTER_OTLP_PROTOCOL,
    OTEL_EXPORTER_OTLP_PROTOCOL_HTTP_PROTOBUF,
};
use opentelemetry_sdk::metrics::{PeriodicReader, SdkMeterProvider};
use opentelemetry_sdk::resource::{EnvResourceDetector, TelemetryResourceDetector};
use opentelemetry_sdk::Resource;

use crate::error::{Error, Result};

/// Export interval when `OTEL_METRIC_EXPORT_INTERVAL` is not set: a
/// dashboard of buses wants more than the SDK's 60 s.
const DEFAULT_INTERVAL: Duration = Duration::from_secs(5);

/// The provider [`init_from_env`] installed, for [`shutdown`].
static PROVIDER: Mutex<Option<SdkMeterProvider>> = Mutex::new(None);

/// A set, non-blank environment variable.
fn env_var(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|v| v.trim().to_owned())
        .filter(|v| !v.is_empty())
}

/// Whether the environment (`var`) asks for metrics to be exported.
fn requested(var: impl Fn(&str) -> Option<String>) -> Result<bool> {
    if var("OTEL_SDK_DISABLED").is_some_and(|v| v.eq_ignore_ascii_case("true")) {
        return Ok(false);
    }
    match var("OTEL_METRICS_EXPORTER").as_deref() {
        None | Some("otlp") => {}
        Some("none") => return Ok(false),
        Some(other) => {
            return Err(Error::Config(format!(
                "OTEL_METRICS_EXPORTER={other:?}: hilspy exports metrics with \"otlp\" or \"none\""
            )))
        }
    }
    for name in [
        OTEL_EXPORTER_OTLP_METRICS_PROTOCOL,
        OTEL_EXPORTER_OTLP_PROTOCOL,
    ] {
        if let Some(p) = var(name) {
            if p != OTEL_EXPORTER_OTLP_PROTOCOL_HTTP_PROTOBUF {
                return Err(Error::Config(format!(
                    "{name}={p:?}: hilspy exports OTLP over HTTP \
                     ({OTEL_EXPORTER_OTLP_PROTOCOL_HTTP_PROTOBUF:?})"
                )));
            }
            break;
        }
    }
    Ok([
        OTEL_EXPORTER_OTLP_METRICS_ENDPOINT,
        OTEL_EXPORTER_OTLP_ENDPOINT,
    ]
    .into_iter()
    .any(|name| var(name).is_some()))
}

/// Install a meter provider exporting over OTLP/HTTP (protobuf) if the
/// environment asks for it, once per process; later calls return the
/// first's outcome. [`crate::Channel::open`] calls it.
///
/// Exports when `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` or
/// `OTEL_EXPORTER_OTLP_ENDPOINT` is set (the latter gets `/v1/metrics`
/// appended), unless `OTEL_SDK_DISABLED=true` or
/// `OTEL_METRICS_EXPORTER=none`. `OTEL_EXPORTER_OTLP_HEADERS`,
/// `OTEL_EXPORTER_OTLP_TIMEOUT`, `OTEL_METRIC_EXPORT_INTERVAL` (default here
/// 5000 ms), `OTEL_SERVICE_NAME` (default `hilspy`) and
/// `OTEL_RESOURCE_ATTRIBUTES` apply as usual; `host.name` and
/// `service.instance.id` (`host:pid`) are filled in. The host's load is
/// exported too ([`super::register_host_metrics`]).
///
/// Returns whether metrics are exported; a variable hilspy cannot honour
/// (gRPC, another exporter) is a `Config` error.
pub fn init_from_env() -> Result<bool> {
    static INIT: OnceLock<std::result::Result<bool, String>> = OnceLock::new();
    INIT.get_or_init(|| install().map_err(|e| e.to_string()))
        .clone()
        .map_err(|e| Error::Config(format!("telemetry: {e}")))
}

fn install() -> Result<bool> {
    if !requested(env_var)? {
        return Ok(false);
    }
    let exporter = MetricExporter::builder()
        .with_http()
        .build()
        .map_err(|e| Error::Config(e.to_string()))?;
    let mut reader = PeriodicReader::builder(exporter);
    if env_var("OTEL_METRIC_EXPORT_INTERVAL").is_none() {
        reader = reader.with_interval(DEFAULT_INTERVAL);
    }
    let provider = SdkMeterProvider::builder()
        .with_resource(resource(env_var("OTEL_SERVICE_NAME")))
        .with_reader(reader.build())
        .build();
    opentelemetry::global::set_meter_provider(provider.clone());
    super::register_host_metrics(&opentelemetry::global::meter(super::SCOPE));
    *PROVIDER.lock().unwrap() = Some(provider);
    Ok(true)
}

/// hilspy's defaults, then `OTEL_RESOURCE_ATTRIBUTES`, then
/// `OTEL_SERVICE_NAME`, given as `service` (the SDK's own detector would
/// default the service to `unknown_service`).
fn resource(service: Option<String>) -> Resource {
    let host = sysinfo::System::host_name().unwrap_or_else(|| "localhost".into());
    let mut builder = Resource::builder_empty()
        .with_attributes([
            KeyValue::new("service.name", "hilspy"),
            KeyValue::new(
                "service.instance.id",
                format!("{host}:{}", std::process::id()),
            ),
            KeyValue::new("host.name", host),
        ])
        .with_detectors(&[
            Box::new(TelemetryResourceDetector),
            Box::new(EnvResourceDetector::new()),
        ]);
    if let Some(name) = service {
        builder = builder.with_service_name(name);
    }
    builder.build()
}

/// Export what is recorded and stop exporting (at exit; the Python package
/// calls it from `atexit`). Idempotent; a no-op when nothing is exported.
pub fn shutdown() {
    if let Some(p) = PROVIDER.lock().unwrap().take() {
        let _ = p.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn ask(vars: &[(&str, &str)]) -> Result<bool> {
        let vars: HashMap<String, String> = vars
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect();
        requested(|name| vars.get(name).cloned())
    }

    #[test]
    fn an_endpoint_turns_exporting_on() {
        const BOTH: &str = OTEL_EXPORTER_OTLP_ENDPOINT;
        assert!(!ask(&[]).unwrap());
        assert!(ask(&[(BOTH, "http://h:9090/api/v1/otlp")]).unwrap());
        assert!(ask(&[(OTEL_EXPORTER_OTLP_METRICS_ENDPOINT, "http://b/v1/metrics")]).unwrap());
        assert!(!ask(&[(BOTH, "x"), ("OTEL_SDK_DISABLED", "TRUE")]).unwrap());
        assert!(!ask(&[(BOTH, "x"), ("OTEL_METRICS_EXPORTER", "none")]).unwrap());
        assert!(ask(&[(BOTH, "x"), ("OTEL_METRICS_EXPORTER", "prometheus")]).is_err());
        let grpc = ask(&[(BOTH, "x"), (OTEL_EXPORTER_OTLP_PROTOCOL, "grpc")]);
        assert!(grpc.unwrap_err().to_string().contains("http/protobuf"));
        // The metrics-specific variable wins.
        let ok = ask(&[
            (BOTH, "x"),
            (OTEL_EXPORTER_OTLP_PROTOCOL, "grpc"),
            (OTEL_EXPORTER_OTLP_METRICS_PROTOCOL, "http/protobuf"),
        ]);
        assert!(ok.unwrap());
    }

    #[test]
    fn the_resource_names_the_service_and_host() {
        let name = |r: &Resource| {
            r.get(&opentelemetry::Key::from_static_str("service.name"))
                .unwrap()
                .to_string()
        };
        let r = resource(None);
        assert_eq!(name(&r), "hilspy");
        let host = r
            .get(&opentelemetry::Key::from_static_str("host.name"))
            .unwrap();
        let instance = r
            .get(&opentelemetry::Key::from_static_str("service.instance.id"))
            .unwrap();
        assert_eq!(
            instance.to_string(),
            format!("{host}:{}", std::process::id())
        );
        assert_eq!(name(&resource(Some("brake-ecu".into()))), "brake-ecu");
    }
}
