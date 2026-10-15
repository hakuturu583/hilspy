use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use super::filter::Filter;
use super::frame::Frame;
use super::periodic::PeriodicTask;
use crate::error::{Error, Result};

/// A bus implementation. Methods take `&self` so one bus can be shared by a
/// receiving thread and periodic senders.
pub trait BusBackend: Send + Sync {
    /// Queue `frame` for transmission.
    fn send(&self, frame: &Frame) -> Result<()>;
    /// The next received frame, or `None` once `timeout` passes (`None`
    /// waits forever).
    fn recv(&self, timeout: Option<Duration>) -> Result<Option<Frame>>;
    /// Replace the receive filters (an empty slice receives everything).
    fn set_filters(&self, filters: &[Filter]) -> Result<()>;
    /// Wake blocked receivers and release the interface.
    fn shutdown(&self) {}
}

/// How to open a [`Bus`].
#[derive(Clone, Debug)]
pub struct BusConfig {
    /// `"socketcan"` or `"virtual"`.
    pub interface: String,
    /// Interface name (`vcan0`) or virtual channel name.
    pub channel: String,
    /// Allow sending and receiving CAN FD frames.
    pub fd: bool,
    /// Receive the frames this bus sends.
    pub receive_own_messages: bool,
    pub filters: Vec<Filter>,
}

impl BusConfig {
    pub fn new(interface: impl Into<String>, channel: impl Into<String>) -> Self {
        Self {
            interface: interface.into(),
            channel: channel.into(),
            fd: false,
            receive_own_messages: false,
            filters: Vec::new(),
        }
    }
}

/// An open CAN bus. Cloning shares the same underlying socket.
#[derive(Clone)]
pub struct Bus {
    backend: Arc<dyn BusBackend>,
    config: Arc<BusConfig>,
    closed: Arc<AtomicBool>,
}

impl Bus {
    pub fn open(config: BusConfig) -> Result<Self> {
        let backend: Arc<dyn BusBackend> = match config.interface.as_str() {
            "virtual" => Arc::new(super::virtual_bus::VirtualBus::open(&config)?),
            #[cfg(target_os = "linux")]
            "socketcan" => Arc::new(super::socketcan_bus::SocketCanBus::open(&config)?),
            #[cfg(not(target_os = "linux"))]
            "socketcan" => {
                return Err(Error::Unsupported(
                    "the socketcan interface is only available on Linux".into(),
                ))
            }
            other => {
                return Err(Error::Unsupported(format!(
                    "unknown CAN interface {other:?} (expected \"socketcan\" or \"virtual\")"
                )))
            }
        };
        Ok(Self::from_backend(backend, config))
    }

    /// Wrap a custom backend.
    pub fn from_backend(backend: Arc<dyn BusBackend>, config: BusConfig) -> Self {
        Self {
            backend,
            config: Arc::new(config),
            closed: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn config(&self) -> &BusConfig {
        &self.config
    }

    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    pub fn send(&self, frame: &Frame) -> Result<()> {
        if self.is_closed() {
            return Err(Error::BusClosed);
        }
        frame.validate()?;
        if frame.fd && !self.config.fd {
            return Err(Error::InvalidFrame(
                "a CAN FD frame on a bus opened without fd=True".into(),
            ));
        }
        self.backend.send(frame)
    }

    pub fn recv(&self, timeout: Option<Duration>) -> Result<Option<Frame>> {
        if self.is_closed() {
            return Err(Error::BusClosed);
        }
        match self.backend.recv(timeout) {
            Err(Error::BusClosed) if self.is_closed() => Err(Error::BusClosed),
            other => other,
        }
    }

    pub fn set_filters(&self, filters: &[Filter]) -> Result<()> {
        if self.is_closed() {
            return Err(Error::BusClosed);
        }
        self.backend.set_filters(filters)
    }

    /// Send `frame` every `period` from a background thread until the task
    /// is stopped or dropped.
    pub fn send_periodic(
        &self,
        frame: Frame,
        period: Duration,
        duration: Option<Duration>,
    ) -> Result<PeriodicTask> {
        PeriodicTask::start(self.clone(), frame, period, duration)
    }

    /// Close the bus: blocked `recv` calls return and later calls fail.
    pub fn shutdown(&self) {
        if !self.closed.swap(true, Ordering::AcqRel) {
            self.backend.shutdown();
        }
    }
}
