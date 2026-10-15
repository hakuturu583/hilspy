//! Rust core of `hilspy`, a hardware-in-the-loop simulation (HILS) I/O library.
//!
//! * [`can`]: CAN / CAN FD frames, buses (Linux SocketCAN incl. `vcan`, and an
//!   in-process virtual bus) and periodic transmission.
//! * [`dbc`]: a DBC database (parsed with the `can-dbc` crate) that encodes
//!   and decodes signals.
//! * [`sbf`]: a streaming Septentrio Binary Format (SBF) decoder (block
//!   bodies parsed with the `libsbf` crate) and encoders for the blocks a
//!   simulated GNSS receiver emits.
//!
//! The crate has no Python dependency; `hilspy-python` is a thin PyO3 layer
//! over it.

pub mod can;
pub mod dbc;
pub mod error;
pub mod sbf;

pub use error::{Error, Result};
