//! Rust core of `hilspy`, a hardware-in-the-loop simulation (HILS) I/O
//! library.
//!
//! A [`Channel`] composes a [`Transport`] (how packets travel: `can://`,
//! `udp://`, `tcp://`, `mem://`) with a [`Codec`] (what they mean: `raw`,
//! `dbc:FILE`, `sbf`, `lidar:MODEL`). Both are pluggable: [`transport::register`] and
//! [`codec::register`] add schemes and codecs, and the Python bindings use
//! them to accept transports and codecs written in Python.
//!
//! The crate has no Python dependency; `hilspy-python` is a thin PyO3 layer
//! over it.

pub mod channel;
pub mod clock;
pub mod codec;
pub mod dbc;
pub mod error;
pub mod lidar;
pub mod message;
mod queue;
pub mod sbf;
mod time;
pub mod transport;
pub mod url;
pub mod value;

pub use channel::{Channel, Periodic, Subscription};
pub use clock::Clock;
pub use codec::Codec;
pub use error::{Error, Result};
pub use message::{CanHeader, Message, Packet};
pub use transport::Transport;
pub use url::{CodecSpec, Endpoint};
pub use value::{Array, Options, Value};
