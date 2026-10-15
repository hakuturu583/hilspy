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

/// The checksum model of a LIN frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinChecksum {
    /// LIN 1.x: over the data bytes only. Diagnostic frames (`0x3C`,
    /// `0x3D`) always use it.
    Classic,
    /// LIN 2.x: over the protected identifier and the data bytes.
    Enhanced,
}

impl LinChecksum {
    /// The model LIN 2.x uses for frame `id`: classic for the diagnostic
    /// frames `0x3C` and `0x3D`, enhanced for the others.
    pub fn for_id(id: u8) -> Self {
        if is_lin_diagnostic(id) {
            Self::Classic
        } else {
            Self::Enhanced
        }
    }

    /// The checksum byte of frame `id` carrying `data`: the inverted eight-bit
    /// sum with carry (each carry added back) of the data bytes, preceded by
    /// the protected identifier for the enhanced model. Diagnostic frames
    /// are always classic.
    pub fn compute(self, id: u8, data: &[u8]) -> u8 {
        let start = match self {
            Self::Enhanced if !is_lin_diagnostic(id) => u16::from(lin_protected_id(id)),
            _ => 0,
        };
        let sum = data.iter().fold(start, |acc, &b| {
            let s = acc + u16::from(b);
            if s > 0xFF {
                s - 0xFF
            } else {
                s
            }
        });
        !(sum as u8)
    }
}

/// Whether `id` is a LIN diagnostic frame (master request `0x3C`, slave
/// response `0x3D`), whose checksum is always classic.
pub fn is_lin_diagnostic(id: u8) -> bool {
    matches!(id, 0x3C | 0x3D)
}

/// The protected identifier of LIN frame `id` (0..=63): the identifier in
/// bits 0-5, parity `P0 = ID0 ^ ID1 ^ ID2 ^ ID4` in bit 6 and
/// `P1 = !(ID1 ^ ID3 ^ ID4 ^ ID5)` in bit 7.
pub fn lin_protected_id(id: u8) -> u8 {
    let id = id & 0x3F;
    let bit = |n: u8| (id >> n) & 1;
    let p0 = bit(0) ^ bit(1) ^ bit(2) ^ bit(4);
    let p1 = 1 ^ bit(1) ^ bit(3) ^ bit(4) ^ bit(5);
    id | (p0 << 6) | (p1 << 7)
}

/// The header of a LIN frame: what the master's header and the response's
/// checksum carry besides the data bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinHeader {
    /// Frame identifier, 0..=63 (without the parity bits).
    pub id: u8,
    /// The checksum model the frame was sent with.
    pub checksum_model: LinChecksum,
    /// The checksum byte as transmitted.
    pub checksum: u8,
}

impl LinHeader {
    /// The protected identifier (identifier and parity bits).
    pub fn protected_id(&self) -> u8 {
        lin_protected_id(self.id)
    }

    /// Whether the checksum matches `data` under the header's model.
    pub fn checksum_ok(&self, data: &[u8]) -> bool {
        self.checksum_model.compute(self.id, data) == self.checksum
    }
}

/// The FlexRay channel(s) a frame is sent on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlexRayChannel {
    A,
    B,
    /// Both channels.
    AB,
}

impl FlexRayChannel {
    /// `"A"`, `"B"` or `"AB"`.
    pub fn name(self) -> &'static str {
        match self {
            Self::A => "A",
            Self::B => "B",
            Self::AB => "AB",
        }
    }

    /// Parse `A`, `B` or `AB` (any case).
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_uppercase().as_str() {
            "A" => Some(Self::A),
            "B" => Some(Self::B),
            "AB" | "A+B" | "BOTH" => Some(Self::AB),
            _ => None,
        }
    }

    /// Whether a frame on `self` is seen on `other` (`AB` shares a channel
    /// with both).
    pub fn overlaps(self, other: Self) -> bool {
        self == Self::AB || other == Self::AB || self == other
    }
}

/// The header of a FlexRay frame: where in the schedule it was sent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlexRayHeader {
    /// Slot identifier, 1..=2047.
    pub slot: u16,
    /// Cycle counter, 0..=63.
    pub cycle: u8,
    pub channel: FlexRayChannel,
}

/// One unit a transport sends or receives: a CAN, LIN or FlexRay frame, a UDP
/// datagram or a chunk of a TCP stream.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Packet {
    /// The payload: a CAN or FlexRay payload, the data bytes of a LIN frame
    /// (its checksum is in the header), a datagram, a chunk.
    pub data: Vec<u8>,
    /// Set for CAN frames.
    pub can: Option<CanHeader>,
    /// Set for LIN frames.
    pub lin: Option<LinHeader>,
    /// Set for FlexRay frames.
    pub flexray: Option<FlexRayHeader>,
    /// Reception time, seconds since the Unix epoch.
    pub timestamp: Option<f64>,
    /// The connection a packet arrived on, for transports with several
    /// (a TCP server): stream codecs keep one decoder per source.
    pub source: Option<u64>,
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
            ..Self::default()
        }
    }

    /// A LIN frame with its checksum computed under `model`.
    pub fn lin(id: u8, model: LinChecksum, data: impl Into<Vec<u8>>) -> Self {
        let data = data.into();
        Self {
            lin: Some(LinHeader {
                id,
                checksum_model: model,
                checksum: model.compute(id, &data),
            }),
            data,
            ..Self::default()
        }
    }

    /// A FlexRay frame.
    pub fn flexray(
        slot: u16,
        cycle: u8,
        channel: FlexRayChannel,
        data: impl Into<Vec<u8>>,
    ) -> Self {
        Self {
            data: data.into(),
            flexray: Some(FlexRayHeader {
                slot,
                cycle,
                channel,
            }),
            ..Self::default()
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

impl Packet {
    /// Check the bus headers: at most one of CAN, LIN and FlexRay; a LIN
    /// identifier of 0..=63 with up to 8 data bytes; a FlexRay slot of
    /// 1..=2047 and cycle of 0..=63 with an even payload of up to 254 bytes
    /// (FlexRay counts its payload in two-byte words). CAN frames are
    /// checked by the CAN transports.
    pub fn check_headers(&self) -> crate::Result<()> {
        let bad = |m: String| Err(crate::Error::InvalidPacket(m));
        let set = [
            self.can.is_some(),
            self.lin.is_some(),
            self.flexray.is_some(),
        ];
        if set.iter().filter(|s| **s).count() > 1 {
            return bad("a packet is a CAN, a LIN or a FlexRay frame, not several".into());
        }
        let len = self.data.len();
        if let Some(h) = &self.lin {
            if h.id > 0x3F {
                return bad(format!("LIN identifier {} is not in 0..=63", h.id));
            }
            if len > 8 {
                return bad(format!("a LIN frame has at most 8 data bytes, not {len}"));
            }
        }
        if let Some(h) = &self.flexray {
            if !(1..=2047).contains(&h.slot) {
                return bad(format!("FlexRay slot {} is not in 1..=2047", h.slot));
            }
            if h.cycle > 63 {
                return bad(format!("FlexRay cycle {} is not in 0..=63", h.cycle));
            }
            if len > 254 || len % 2 != 0 {
                return bad(format!(
                    "a FlexRay payload is an even number of bytes up to 254, not {len}"
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lin_protected_ids() {
        // LIN 2.2A, table of protected identifiers.
        let table = [
            (0x00, 0x80),
            (0x01, 0xC1),
            (0x02, 0x42),
            (0x10, 0x50),
            (0x11, 0x11),
            (0x20, 0x20),
            (0x3C, 0x3C),
            (0x3D, 0x7D),
            (0x3F, 0xBF),
        ];
        for (id, pid) in table {
            assert_eq!(lin_protected_id(id), pid, "id 0x{id:02X}");
        }
    }

    #[test]
    fn lin_checksums() {
        // Classic: !(0x4A + 0x55 + 0x93 + 0xE5 with carries).
        // 0x4A + 0x55 = 0x9F; + 0x93 = 0x132 -> 0x33; + 0xE5 = 0x118 -> 0x19.
        let data = [0x4A, 0x55, 0x93, 0xE5];
        assert_eq!(LinChecksum::Classic.compute(0x10, &data), !0x19);
        // Enhanced adds the protected id 0x50 first: 0x50 + 0x4A = 0x9A;
        // + 0x55 = 0xEF; + 0x93 = 0x182 -> 0x83; + 0xE5 = 0x168 -> 0x69.
        assert_eq!(LinChecksum::Enhanced.compute(0x10, &data), !0x69);
        // Diagnostic frames are classic whatever the model.
        let diag = [0x7F, 0x06, 0xB2, 0x00, 0xFF, 0x7F, 0xFF, 0xFF];
        assert_eq!(
            LinChecksum::Enhanced.compute(0x3C, &diag),
            LinChecksum::Classic.compute(0x3C, &diag)
        );
        assert_eq!(LinChecksum::for_id(0x3D), LinChecksum::Classic);
        assert_eq!(LinChecksum::for_id(0x10), LinChecksum::Enhanced);
        // No data: classic 0xFF, enhanced the inverted protected id.
        assert_eq!(LinChecksum::Classic.compute(0x11, &[]), 0xFF);
        assert_eq!(LinChecksum::Enhanced.compute(0x11, &[]), !0x11);
        let p = Packet::lin(0x10, LinChecksum::Enhanced, data);
        let h = p.lin.unwrap();
        assert_eq!((h.protected_id(), h.checksum), (0x50, !0x69));
        assert!(h.checksum_ok(&p.data));
        assert!(!h.checksum_ok(&[0x4A, 0x55, 0x93, 0xE4]));
    }

    #[test]
    fn checks_headers() {
        assert!(Packet::lin(63, LinChecksum::Classic, vec![0; 8])
            .check_headers()
            .is_ok());
        assert!(Packet::lin(64, LinChecksum::Classic, vec![])
            .check_headers()
            .is_err());
        assert!(Packet::lin(1, LinChecksum::Classic, vec![0; 9])
            .check_headers()
            .is_err());
        let fr = |slot, cycle, len| Packet::flexray(slot, cycle, FlexRayChannel::A, vec![0; len]);
        assert!(fr(1, 63, 254).check_headers().is_ok());
        assert!(fr(0, 0, 2).check_headers().is_err());
        assert!(fr(2048, 0, 2).check_headers().is_err());
        assert!(fr(1, 64, 2).check_headers().is_err());
        assert!(fr(1, 0, 3).check_headers().is_err());
        assert!(fr(1, 0, 256).check_headers().is_err());
        let mut both = Packet::can(1, vec![]);
        both.lin = Packet::lin(1, LinChecksum::Classic, vec![]).lin;
        assert!(both.check_headers().is_err());
        assert!(FlexRayChannel::AB.overlaps(FlexRayChannel::B));
        assert!(!FlexRayChannel::A.overlaps(FlexRayChannel::B));
        assert_eq!(FlexRayChannel::parse("ab"), Some(FlexRayChannel::AB));
    }
}
