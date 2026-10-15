//! Transports: how packets reach a device.
//!
//! | URL | Transport |
//! | --- | --- |
//! | `can://vcan0` | Linux SocketCAN (`?fd`, `?echo`, `?filter=ID:MASK,...`) |
//! | `mem://name` | in-process bus shared by every channel opened on `name` |
//! | `udp://host:port` | UDP datagrams to/from a peer (`?bind=addr:port`) |
//! | `udp://addr:port?listen` | UDP bound to `addr:port`, replying to the last sender |
//! | `tcp://host:port` | TCP client; packets are stream chunks |
//! | `tcp://addr:port?listen` | TCP server; sends go to every connected client |
//! | `file://path` | replay a recording (`?follow`), or record one (`?write`, `?append`) |
//! | `serial://PORT` | a serial port (`?baud=`, `data_bits`, `parity`, `stop_bits`, `flow`); packets are stream chunks |
//!
//! Other schemes are added with [`register`].

pub(crate) mod can_frame;
#[cfg(target_os = "linux")]
mod can_socket;
mod file;
mod mem;
mod serial;
mod tcp;
mod udp;

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};
use std::time::Duration;

use crate::error::{Error, Result};
use crate::message::Packet;
use crate::url::Endpoint;

/// A packet transport. Methods take `&self`: one transport serves a
/// receiving thread and periodic senders at once.
pub trait Transport: Send + Sync {
    fn send(&self, packet: &Packet) -> Result<()>;
    /// The next packet, or `None` once `timeout` passes (`None`: wait
    /// forever). After `close`, `Err(Error::Closed)`.
    fn recv(&self, timeout: Option<Duration>) -> Result<Option<Packet>>;
    /// Release the transport and wake blocked receivers.
    fn close(&self) {}
}

/// Creates a transport from a parsed URL.
pub type Factory = Arc<dyn Fn(Endpoint) -> Result<Arc<dyn Transport>> + Send + Sync>;

fn registry() -> &'static RwLock<HashMap<String, Factory>> {
    static REGISTRY: OnceLock<RwLock<HashMap<String, Factory>>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let mut m: HashMap<String, Factory> = HashMap::new();
        m.insert(
            "file".into(),
            Arc::new(|e| Ok(Arc::new(file::FileTransport::open(e)?) as Arc<dyn Transport>)),
        );
        m.insert(
            "mem".into(),
            Arc::new(|e| Ok(Arc::new(mem::Mem::open(e)?) as Arc<dyn Transport>)),
        );
        m.insert(
            "udp".into(),
            Arc::new(|e| Ok(Arc::new(udp::Udp::open(e)?) as Arc<dyn Transport>)),
        );
        m.insert(
            "serial".into(),
            Arc::new(|e| Ok(Arc::new(serial::Serial::open(e)?) as Arc<dyn Transport>)),
        );
        m.insert(
            "tcp".into(),
            Arc::new(|e| Ok(Arc::new(tcp::Tcp::open(e)?) as Arc<dyn Transport>)),
        );
        #[cfg(target_os = "linux")]
        m.insert(
            "can".into(),
            Arc::new(|e| Ok(Arc::new(can_socket::SocketCan::open(e)?) as Arc<dyn Transport>)),
        );
        #[cfg(not(target_os = "linux"))]
        m.insert(
            "can".into(),
            Arc::new(|_| {
                Err(Error::Config(
                    "can:// (SocketCAN) is only available on Linux; mem:// works everywhere".into(),
                ))
            }),
        );
        RwLock::new(m)
    })
}

/// Add or replace the transport for a URL scheme.
pub fn register(scheme: &str, factory: Factory) {
    registry()
        .write()
        .unwrap()
        .insert(scheme.to_ascii_lowercase(), factory);
}

/// Registered schemes, sorted.
pub fn schemes() -> Vec<String> {
    let mut s: Vec<_> = registry().read().unwrap().keys().cloned().collect();
    s.sort();
    s
}

/// Open the transport a URL names.
pub fn open(url: &str) -> Result<Arc<dyn Transport>> {
    let endpoint = Endpoint::parse(url)?;
    let factory = registry()
        .read()
        .unwrap()
        .get(&endpoint.scheme)
        .cloned()
        .ok_or_else(|| {
            Error::Config(format!(
                "unknown transport {:?} (known: {})",
                endpoint.scheme,
                schemes().join(", ")
            ))
        })?;
    factory(endpoint)
}
