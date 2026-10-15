use std::fmt;

/// Errors raised by `hilspy-core`.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An operating-system error (socket, file).
    #[error("{0}")]
    Io(#[from] std::io::Error),
    /// A frame that cannot exist on the bus (identifier or length out of range).
    #[error("invalid CAN frame: {0}")]
    InvalidFrame(String),
    /// The bus was shut down.
    #[error("the bus is shut down")]
    BusClosed,
    /// An interface name the backend does not know.
    #[error("{0}")]
    Unsupported(String),
    /// A DBC file that does not parse or is inconsistent.
    #[error("DBC parse error: {0}")]
    DbcParse(String),
    /// A message or signal not in the database.
    #[error("{0}")]
    NotFound(String),
    /// Signal values that cannot be encoded.
    #[error("encode error: {0}")]
    Encode(String),
    /// Payload that cannot be decoded.
    #[error("decode error: {0}")]
    Decode(String),
    /// Invalid SBF input.
    #[error("SBF error: {0}")]
    Sbf(String),
}

impl Error {
    pub(crate) fn encode(msg: impl fmt::Display) -> Self {
        Error::Encode(msg.to_string())
    }

    pub(crate) fn decode(msg: impl fmt::Display) -> Self {
        Error::Decode(msg.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;
