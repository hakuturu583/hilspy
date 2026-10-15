use crate::error::{Error, Result};

/// Largest standard (11-bit) identifier.
pub const SFF_MASK: u32 = 0x7FF;
/// Largest extended (29-bit) identifier.
pub const EFF_MASK: u32 = 0x1FFF_FFFF;
/// Payload length of a classic CAN frame.
pub const CAN_MAX_DLEN: usize = 8;
/// Payload length of a CAN FD frame.
pub const CAN_FD_MAX_DLEN: usize = 64;

/// One CAN or CAN FD frame.
///
/// Construct it with [`Frame::new`] (data), [`Frame::new_fd`] or
/// [`Frame::new_remote`] so the identifier and length are checked; a frame
/// that passed those checks can be sent on any bus that supports its kind.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    /// Arbitration identifier, without flag bits.
    pub id: u32,
    /// 29-bit identifier.
    pub extended: bool,
    /// Remote transmission request (classic CAN only, no data).
    pub remote: bool,
    /// Error frame reported by the controller (`id` holds the error class).
    pub error: bool,
    /// CAN FD frame.
    pub fd: bool,
    /// CAN FD bit rate switch.
    pub brs: bool,
    /// CAN FD error state indicator.
    pub esi: bool,
    /// Requested length of a remote frame; the data length otherwise.
    pub dlc: u8,
    /// Payload.
    pub data: Vec<u8>,
    /// Reception time, seconds since the Unix epoch (`None` for frames not
    /// received from a bus).
    pub timestamp: Option<f64>,
}

impl Frame {
    /// A classic CAN data frame.
    pub fn new(id: u32, data: &[u8], extended: bool) -> Result<Self> {
        check_id(id, extended)?;
        if data.len() > CAN_MAX_DLEN {
            return Err(Error::InvalidFrame(format!(
                "a classic CAN frame carries at most {CAN_MAX_DLEN} bytes, got {} (use a CAN FD frame)",
                data.len()
            )));
        }
        Ok(Self {
            id,
            extended,
            remote: false,
            error: false,
            fd: false,
            brs: false,
            esi: false,
            dlc: data.len() as u8,
            data: data.to_vec(),
            timestamp: None,
        })
    }

    /// A CAN FD data frame. `data` must have a valid CAN FD length
    /// (0-8, 12, 16, 20, 24, 32, 48 or 64); see [`Frame::fd_padded_len`].
    pub fn new_fd(id: u32, data: &[u8], extended: bool, brs: bool) -> Result<Self> {
        check_id(id, extended)?;
        if !is_valid_fd_len(data.len()) {
            return Err(Error::InvalidFrame(format!(
                "{} bytes is not a CAN FD length (0-8, 12, 16, 20, 24, 32, 48, 64)",
                data.len()
            )));
        }
        Ok(Self {
            id,
            extended,
            remote: false,
            error: false,
            fd: true,
            brs,
            esi: false,
            dlc: data.len() as u8,
            data: data.to_vec(),
            timestamp: None,
        })
    }

    /// A classic CAN remote frame requesting `dlc` bytes.
    pub fn new_remote(id: u32, dlc: u8, extended: bool) -> Result<Self> {
        check_id(id, extended)?;
        if dlc as usize > CAN_MAX_DLEN {
            return Err(Error::InvalidFrame(format!(
                "a remote frame requests at most {CAN_MAX_DLEN} bytes, got {dlc}"
            )));
        }
        Ok(Self {
            id,
            extended,
            remote: true,
            error: false,
            fd: false,
            brs: false,
            esi: false,
            dlc,
            data: Vec::new(),
            timestamp: None,
        })
    }

    /// The smallest valid CAN FD payload length that holds `len` bytes.
    pub fn fd_padded_len(len: usize) -> Option<usize> {
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32, 48, 64]
            .into_iter()
            .find(|&l| l >= len)
    }

    /// Check the invariants the constructors establish (for frames whose
    /// fields were changed afterwards).
    pub fn validate(&self) -> Result<()> {
        if self.error {
            return Err(Error::InvalidFrame("error frames cannot be sent".into()));
        }
        check_id(self.id, self.extended)?;
        if self.fd {
            if self.remote {
                return Err(Error::InvalidFrame("CAN FD has no remote frames".into()));
            }
            if !is_valid_fd_len(self.data.len()) {
                return Err(Error::InvalidFrame(format!(
                    "{} bytes is not a CAN FD length",
                    self.data.len()
                )));
            }
        } else if self.remote {
            if !self.data.is_empty() || self.dlc as usize > CAN_MAX_DLEN {
                return Err(Error::InvalidFrame(
                    "a remote frame has no data and a DLC of at most 8".into(),
                ));
            }
        } else if self.data.len() > CAN_MAX_DLEN {
            return Err(Error::InvalidFrame(format!(
                "a classic CAN frame carries at most {CAN_MAX_DLEN} bytes, got {}",
                self.data.len()
            )));
        }
        Ok(())
    }
}

pub(crate) fn is_valid_fd_len(len: usize) -> bool {
    len <= 8 || matches!(len, 12 | 16 | 20 | 24 | 32 | 48 | 64)
}

fn check_id(id: u32, extended: bool) -> Result<()> {
    let max = if extended { EFF_MASK } else { SFF_MASK };
    if id > max {
        return Err(Error::InvalidFrame(format!(
            "identifier 0x{id:X} exceeds the {} maximum 0x{max:X}",
            if extended { "29-bit" } else { "11-bit" }
        )));
    }
    Ok(())
}

pub(crate) fn now_timestamp() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifiers_are_range_checked() {
        assert!(Frame::new(0x7FF, &[], false).is_ok());
        assert!(Frame::new(0x800, &[], false).is_err());
        assert!(Frame::new(0x1FFF_FFFF, &[], true).is_ok());
        assert!(Frame::new(0x2000_0000, &[], true).is_err());
    }

    #[test]
    fn lengths_are_checked() {
        assert!(Frame::new(1, &[0; 9], false).is_err());
        assert!(Frame::new_fd(1, &[0; 12], false, true).is_ok());
        assert!(Frame::new_fd(1, &[0; 13], false, true).is_err());
        assert_eq!(Frame::fd_padded_len(13), Some(16));
        assert_eq!(Frame::fd_padded_len(65), None);
        assert!(Frame::new_remote(1, 9, false).is_err());
    }
}
