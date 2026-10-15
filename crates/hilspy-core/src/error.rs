use std::fmt;

/// Errors raised by `hilspy-core`.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An operating-system error (socket, file).
    #[error("{0}")]
    Io(#[from] std::io::Error),
    /// A bad URL, codec specification or option, or an unknown scheme/codec.
    #[error("{0}")]
    Config(String),
    /// A packet the transport cannot carry (a CAN identifier or length out of
    /// range, a CAN transport given a packet without an identifier, ...).
    #[error("invalid packet: {0}")]
    InvalidPacket(String),
    /// The channel or transport was closed.
    #[error("the channel is closed")]
    Closed,
    /// A DBC file that does not parse or is inconsistent.
    #[error("DBC parse error: {0}")]
    DbcParse(String),
    /// A message or field the codec does not know.
    #[error("{0}")]
    NotFound(String),
    /// A message that cannot be encoded.
    #[error("encode error: {0}")]
    Encode(String),
    /// Data that cannot be decoded.
    #[error("decode error: {0}")]
    Decode(String),
    /// Invalid SBF input.
    #[error("SBF error: {0}")]
    Sbf(String),
    /// An error raised by a plugin (a Python transport or codec); the
    /// bindings re-raise the original exception.
    #[error("{0}")]
    External(Box<dyn std::error::Error + Send + Sync>),
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
