//! CAN frame rules and receive filters shared by `can://` and `mem://`.

use crate::error::{Error, Result};
use crate::message::{CanHeader, Packet};

/// Largest standard (11-bit) identifier.
pub const SFF_MASK: u32 = 0x7FF;
/// Largest extended (29-bit) identifier.
pub const EFF_MASK: u32 = 0x1FFF_FFFF;
/// Payload length of a classic CAN frame.
pub const CAN_MAX_DLEN: usize = 8;

pub(crate) fn is_valid_fd_len(len: usize) -> bool {
    len <= 8 || matches!(len, 12 | 16 | 20 | 24 | 32 | 48 | 64)
}

/// The smallest valid CAN FD payload length that holds `len` bytes.
pub fn fd_padded_len(len: usize) -> Option<usize> {
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32, 48, 64]
        .into_iter()
        .find(|&l| l >= len)
}

/// Check that a packet is a valid CAN frame for a bus with or without CAN
/// FD, and return its header (FD set when the payload needs it).
pub(crate) fn check(packet: &Packet, fd_bus: bool) -> Result<CanHeader> {
    let mut h = packet
        .can
        .ok_or_else(|| Error::InvalidPacket("a CAN transport needs a message with an id".into()))?;
    let max = if h.extended { EFF_MASK } else { SFF_MASK };
    if h.id > max {
        return Err(Error::InvalidPacket(format!(
            "identifier 0x{:X} exceeds the {} maximum 0x{max:X}",
            h.id,
            if h.extended { "29-bit" } else { "11-bit" }
        )));
    }
    let len = packet.data.len();
    if len > CAN_MAX_DLEN {
        h.fd = true;
    }
    if h.fd {
        if !fd_bus {
            return Err(Error::InvalidPacket(format!(
                "a {len}-byte (CAN FD) frame on a bus opened without ?fd"
            )));
        }
        if !is_valid_fd_len(len) {
            return Err(Error::InvalidPacket(format!(
                "{len} bytes is not a CAN FD length (0-8, 12, 16, 20, 24, 32, 48, 64)"
            )));
        }
    } else if h.brs {
        return Err(Error::InvalidPacket(
            "bit rate switch needs a CAN FD frame".into(),
        ));
    }
    Ok(h)
}

/// A receive filter: `id & mask == filter.id & mask`, for one identifier kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Filter {
    pub id: u32,
    pub mask: u32,
    pub extended: bool,
}

impl Filter {
    /// `ID[:MASK]` items separated by commas (hex with 0x, or decimal). An
    /// ID above 0x7FF selects 29-bit identifiers; the mask defaults to an
    /// exact match.
    pub(crate) fn parse_list(spec: &str) -> Result<Vec<Filter>> {
        spec.split(',')
            .filter(|s| !s.is_empty())
            .map(|item| {
                let (id, mask) = item.split_once(':').unwrap_or((item, ""));
                let id = parse_u32(id)?;
                let extended = id > SFF_MASK;
                let mask = if mask.is_empty() {
                    if extended {
                        EFF_MASK
                    } else {
                        SFF_MASK
                    }
                } else {
                    parse_u32(mask)?
                };
                Ok(Filter { id, mask, extended })
            })
            .collect()
    }

    pub(crate) fn matches(&self, h: &CanHeader) -> bool {
        h.extended == self.extended && h.id & self.mask == self.id & self.mask
    }

    pub(crate) fn accepts(filters: &[Filter], packet: &Packet) -> bool {
        match &packet.can {
            None => true,
            Some(h) => filters.is_empty() || filters.iter().any(|f| f.matches(h)),
        }
    }
}

fn parse_u32(s: &str) -> Result<u32> {
    let s = s.trim();
    let r = match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        Some(hex) => u32::from_str_radix(hex, 16),
        None => s.parse(),
    };
    r.map_err(|_| Error::Config(format!("filter: {s:?} is not a number")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checks_frames() {
        assert!(check(&Packet::can(0x7FF, vec![]), false).is_ok());
        let mut p = Packet::can(0x800, vec![]);
        p.can.as_mut().unwrap().extended = false;
        assert!(check(&p, false).is_err());
        assert!(check(&Packet::can(0x1FFF_FFFF, vec![]), false).is_ok());
        assert!(check(&Packet::can(0x2000_0000, vec![]), false).is_err());
        assert!(check(&Packet::can(1, vec![0; 9]), true).is_err());
        assert!(check(&Packet::can(1, vec![0; 12]), false).is_err());
        assert!(check(&Packet::can(1, vec![0; 12]), true).unwrap().fd);
        assert!(check(&Packet::bytes(vec![1]), true).is_err());
        assert_eq!(fd_padded_len(13), Some(16));
    }

    #[test]
    fn parses_filters() {
        let f = Filter::parse_list("0x100:0x700,0x18FF0001").unwrap();
        assert_eq!(
            f[0],
            Filter {
                id: 0x100,
                mask: 0x700,
                extended: false
            }
        );
        assert_eq!(
            f[1],
            Filter {
                id: 0x18FF_0001,
                mask: EFF_MASK,
                extended: true
            }
        );
        assert!(Filter::parse_list("zz").is_err());
        assert!(Filter::accepts(&f, &Packet::can(0x1AB, vec![])));
        assert!(!Filter::accepts(&f, &Packet::can(0x2AB, vec![])));
        assert!(Filter::accepts(&f, &Packet::bytes(vec![])));
    }
}
