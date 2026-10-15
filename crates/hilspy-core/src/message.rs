//! What crosses a transport ([`Packet`]) and what a codec makes of it
//! ([`Message`]).

use crate::value::Value;

/// The CAN header of a packet carried by a CAN transport.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CanHeader {
    /// Arbitration identifier, without flag bits.
    pub id: u32,
    /// 29-bit identifier.
    pub extended: bool,
    /// CAN FD frame.
    pub fd: bool,
    /// CAN FD bit rate switch.
    pub brs: bool,
}

/// One unit a transport sends or receives: a CAN frame, a UDP datagram or a
/// chunk of a TCP stream.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Packet {
    pub data: Vec<u8>,
    /// Set for CAN frames.
    pub can: Option<CanHeader>,
    /// Reception time, seconds since the Unix epoch.
    pub timestamp: Option<f64>,
}

impl Packet {
    pub fn bytes(data: impl Into<Vec<u8>>) -> Self {
        Self {
            data: data.into(),
            ..Self::default()
        }
    }

    pub fn can(id: u32, data: impl Into<Vec<u8>>) -> Self {
        Self {
            data: data.into(),
            can: Some(CanHeader {
                id,
                extended: id > crate::transport::can_frame::SFF_MASK,
                fd: false,
                brs: false,
            }),
            timestamp: None,
        }
    }
}

/// A decoded (or to-be-encoded) message: a name and named fields, with the
/// packet it came from. A message without a name is raw: sending it sends
/// its packet as is.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Message {
    pub name: Option<String>,
    pub fields: Vec<(String, Value)>,
    pub packet: Option<Packet>,
}

impl Message {
    pub fn new(name: impl Into<String>, fields: Vec<(String, Value)>) -> Self {
        Self {
            name: Some(name.into()),
            fields,
            packet: None,
        }
    }

    pub fn raw(packet: Packet) -> Self {
        Self {
            name: None,
            fields: Vec::new(),
            packet: Some(packet),
        }
    }

    pub fn field(&self, name: &str) -> Option<&Value> {
        self.fields.iter().find(|(n, _)| n == name).map(|(_, v)| v)
    }

    pub fn timestamp(&self) -> Option<f64> {
        self.packet.as_ref().and_then(|p| p.timestamp)
    }
}

pub(crate) fn now_timestamp() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}
