//! Codecs: how packets become named messages and back.
//!
//! | Spec | Codec |
//! | --- | --- |
//! | `raw` | none: every packet is a nameless message carrying its data |
//! | `dbc:FILE` | CAN frames <-> DBC messages (options `strict`, `scaling`, `decode_choices`, `e2e`) |
//! | `fibex:FILE` | CAN frames <-> the frames of a FIBEX channel (option `channel`, and the `dbc` options) |
//! | `sbf` | a Septentrio Binary Format byte stream <-> SBF blocks |
//! | `nmea` | NMEA 0183 sentences <-> messages (`GGA`, `RMC`, ...) |
//! | `ubx` | u-blox UBX frames <-> messages (`NAV-PVT`, ...) |
//! | `rtcm` | RTCM 3 frames <-> messages (`1005`, `1077`, ...) |
//! | `lidar:MODEL` | LiDAR data packets <-> point clouds (Hesai, Velodyne, or a profile file) |
//!
//! Other codecs are added with [`register`].

mod dbc;
mod e2e;
mod fibex;
mod lidar;
mod nmea;
mod raw;
mod rtcm;
mod sbf;
mod ubx;

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use crate::error::{Error, Result};
use crate::message::{Message, Packet};
use crate::url::CodecSpec;
use crate::value::Options;

/// Turns messages into packets and packets into messages. A decoder may keep
/// state across packets (a byte stream split across chunks).
pub trait Codec: Send {
    /// Packets for a named message (nameless messages never reach a codec:
    /// the channel sends their packet as is).
    fn encode(&mut self, message: &Message) -> Result<Vec<Packet>>;
    /// The messages a received packet completes (none, one or several).
    fn decode(&mut self, packet: Packet) -> Result<Vec<Message>>;
    /// Whether the channel sends the packets of one message spaced as their
    /// timestamps are (a LiDAR revolution streamed over its duration) rather
    /// than at once.
    fn paced(&self) -> bool {
        false
    }
}

/// Creates a codec from a parsed specification.
pub type Factory = Arc<dyn Fn(CodecSpec) -> Result<Box<dyn Codec>> + Send + Sync>;

fn registry() -> &'static RwLock<HashMap<String, Factory>> {
    static REGISTRY: OnceLock<RwLock<HashMap<String, Factory>>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let mut m: HashMap<String, Factory> = HashMap::new();
        m.insert(
            "raw".into(),
            Arc::new(|s| Ok(Box::new(raw::Raw::open(s)?) as Box<dyn Codec>)),
        );
        m.insert(
            "dbc".into(),
            Arc::new(|s| Ok(Box::new(dbc::Dbc::open(s)?) as Box<dyn Codec>)),
        );
        m.insert(
            "fibex".into(),
            Arc::new(|s| Ok(Box::new(fibex::open(s)?) as Box<dyn Codec>)),
        );
        m.insert(
            "lidar".into(),
            Arc::new(|s| Ok(Box::new(lidar::Lidar::open(s)?) as Box<dyn Codec>)),
        );
        m.insert(
            "nmea".into(),
            Arc::new(|s| Ok(Box::new(nmea::Nmea::open(s)?) as Box<dyn Codec>)),
        );
        m.insert(
            "ubx".into(),
            Arc::new(|s| Ok(Box::new(ubx::Ubx::open(s)?) as Box<dyn Codec>)),
        );
        m.insert(
            "rtcm".into(),
            Arc::new(|s| Ok(Box::new(rtcm::Rtcm::open(s)?) as Box<dyn Codec>)),
        );
        m.insert(
            "sbf".into(),
            Arc::new(|s| Ok(Box::new(sbf::Sbf::open(s)?) as Box<dyn Codec>)),
        );
        RwLock::new(m)
    })
}

/// Add or replace the codec for a name.
pub fn register(name: &str, factory: Factory) {
    registry()
        .write()
        .unwrap()
        .insert(name.to_ascii_lowercase(), factory);
}

/// Registered codec names, sorted.
pub fn names() -> Vec<String> {
    let mut n: Vec<_> = registry().read().unwrap().keys().cloned().collect();
    n.sort();
    n
}

/// Open the codec a specification names (`raw`, `sbf`, `dbc:FILE`, ...).
pub fn open(spec: &str, options: Options) -> Result<Box<dyn Codec>> {
    let spec = CodecSpec::parse(spec, options)?;
    let factory = registry()
        .read()
        .unwrap()
        .get(&spec.name)
        .cloned()
        .ok_or_else(|| {
            Error::Config(format!(
                "unknown codec {:?} (known: {})",
                spec.name,
                names().join(", ")
            ))
        })?;
    factory(spec)
}
