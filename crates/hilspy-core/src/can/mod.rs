//! CAN / CAN FD frames and buses.
//!
//! A [`Bus`] is opened by interface name:
//!
//! * `"socketcan"` (Linux only): a raw SocketCAN socket on a kernel interface
//!   such as `can0` or `vcan0`.
//! * `"virtual"`: an in-process bus. Every `Bus` opened on the same channel
//!   name in the same process sees the others' frames, without a kernel
//!   interface; for tests and for wiring simulated ECUs together.

mod bus;
mod filter;
mod frame;
mod periodic;
#[cfg(target_os = "linux")]
mod socketcan_bus;
mod virtual_bus;

pub use bus::{Bus, BusBackend, BusConfig};
pub use filter::Filter;
pub use frame::{Frame, CAN_FD_MAX_DLEN, CAN_MAX_DLEN, EFF_MASK, SFF_MASK};
pub use periodic::PeriodicTask;
