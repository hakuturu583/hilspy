//! Linux SocketCAN (`can0`, `vcan0`, ...).

use std::io::ErrorKind;
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use socketcan::{
    CanAnyFrame, CanDataFrame, CanFdFrame, CanFdSocket, CanFilter, CanFrame, CanRemoteFrame,
    CanSocket, EmbeddedFrame, ExtendedId, Id, Socket, SocketOptions, StandardId,
};

use socketcan::id::FdFlags;

use super::bus::{BusBackend, BusConfig};
use super::filter::Filter;
use super::frame::{now_timestamp, Frame};
use crate::error::{Error, Result};

/// `CAN_EFF_FLAG`: the identifier-kind bit filters match on.
const EFF_FLAG: u32 = 0x8000_0000;

/// Run `$body` with `$s` bound to whichever socket the bus holds.
macro_rules! with_sock {
    ($sock:expr, $s:ident => $body:expr) => {
        match $sock {
            Sock::Classic($s) => $body,
            Sock::Fd($s) => $body,
        }
    };
}

enum Sock {
    Classic(CanSocket),
    Fd(CanFdSocket),
}

pub(crate) struct SocketCanBus {
    sock: Sock,
    closed: AtomicBool,
}

/// How often a receiver waiting without a timeout checks for `shutdown`.
const SHUTDOWN_POLL: Duration = Duration::from_millis(100);

impl SocketCanBus {
    pub(crate) fn open(config: &BusConfig) -> Result<Self> {
        let sock = if config.fd {
            Sock::Fd(CanFdSocket::open(&config.channel).map_err(|e| open_error(config, e))?)
        } else {
            Sock::Classic(CanSocket::open(&config.channel).map_err(|e| open_error(config, e))?)
        };
        with_sock!(&sock, s => s.set_recv_own_msgs(config.receive_own_messages))?;
        let bus = Self {
            sock,
            closed: AtomicBool::new(false),
        };
        bus.set_filters(&config.filters)?;
        Ok(bus)
    }

    fn raw_fd(&self) -> i32 {
        with_sock!(&self.sock, s => s.as_raw_fd())
    }
}

fn open_error(config: &BusConfig, e: std::io::Error) -> Error {
    Error::Io(std::io::Error::new(
        e.kind(),
        format!(
            "cannot open SocketCAN interface {:?}: {e} (for a virtual interface: \
             `sudo modprobe vcan && sudo ip link add dev {0} type vcan && sudo ip link set up {0}`)",
            config.channel
        ),
    ))
}

fn embedded_id(id: u32, extended: bool) -> Result<Id> {
    let id = if extended {
        ExtendedId::new(id).map(Id::Extended)
    } else {
        u16::try_from(id)
            .ok()
            .and_then(StandardId::new)
            .map(Id::Standard)
    };
    id.ok_or_else(|| Error::InvalidFrame("identifier out of range".into()))
}

fn invalid(what: &str) -> Error {
    Error::InvalidFrame(format!("SocketCAN rejected the {what}"))
}

fn from_embedded<F: EmbeddedFrame>(f: &F) -> Frame {
    let (id, extended) = match f.id() {
        Id::Standard(id) => (u32::from(id.as_raw()), false),
        Id::Extended(id) => (id.as_raw(), true),
    };
    Frame {
        id,
        extended,
        remote: f.is_remote_frame(),
        error: false,
        fd: false,
        brs: false,
        esi: false,
        dlc: f.dlc() as u8,
        data: if f.is_remote_frame() {
            Vec::new()
        } else {
            f.data().to_vec()
        },
        timestamp: Some(now_timestamp()),
    }
}

fn from_classic(frame: CanFrame) -> Frame {
    match frame {
        CanFrame::Data(f) => from_embedded(&f),
        CanFrame::Remote(f) => from_embedded(&f),
        CanFrame::Error(f) => error_frame(f.error_bits(), f.data()),
    }
}

fn error_frame(bits: u32, data: &[u8]) -> Frame {
    Frame {
        id: bits,
        extended: false,
        remote: false,
        error: true,
        fd: false,
        brs: false,
        esi: false,
        dlc: data.len() as u8,
        data: data.to_vec(),
        timestamp: Some(now_timestamp()),
    }
}

fn from_any(frame: CanAnyFrame) -> Frame {
    match frame {
        CanAnyFrame::Normal(f) => from_embedded(&f),
        CanAnyFrame::Remote(f) => from_embedded(&f),
        CanAnyFrame::Error(f) => error_frame(f.error_bits(), f.data()),
        CanAnyFrame::Fd(f) => {
            let mut out = from_embedded(&f);
            out.fd = true;
            out.brs = f.is_brs();
            out.esi = f.is_esi();
            out
        }
    }
}

fn to_classic(frame: &Frame) -> Result<CanFrame> {
    let id = embedded_id(frame.id, frame.extended)?;
    if frame.remote {
        CanRemoteFrame::new_remote(id, frame.dlc as usize)
            .map(CanFrame::Remote)
            .ok_or_else(|| invalid("remote frame"))
    } else {
        CanDataFrame::new(id, &frame.data)
            .map(CanFrame::Data)
            .ok_or_else(|| invalid("data frame"))
    }
}

fn to_fd(frame: &Frame) -> Result<CanFdFrame> {
    let mut flags = FdFlags::empty();
    if frame.brs {
        flags |= FdFlags::BRS;
    }
    if frame.esi {
        flags |= FdFlags::ESI;
    }
    CanFdFrame::with_flags(embedded_id(frame.id, frame.extended)?, &frame.data, flags)
        .ok_or_else(|| invalid("CAN FD frame"))
}

fn kernel_filter(f: &Filter) -> CanFilter {
    let (mut id, mut mask) = (f.id, f.mask);
    if let Some(extended) = f.extended {
        mask |= EFF_FLAG;
        if extended {
            id |= EFF_FLAG;
        }
    }
    CanFilter::new(id, mask)
}

/// Wait until the socket is readable; `false` on timeout.
fn wait_readable(fd: i32, timeout: Duration) -> Result<bool> {
    let timeout_ms = timeout.as_millis().min(i32::MAX as u128) as i32;
    let mut pfd = libc::pollfd {
        fd,
        events: libc::POLLIN,
        revents: 0,
    };
    loop {
        // SAFETY: one valid pollfd for the duration of the call.
        let n = unsafe { libc::poll(&mut pfd, 1, timeout_ms) };
        if n < 0 {
            let err = std::io::Error::last_os_error();
            if err.kind() == ErrorKind::Interrupted {
                continue;
            }
            return Err(err.into());
        }
        return Ok(n > 0);
    }
}

impl BusBackend for SocketCanBus {
    fn send(&self, frame: &Frame) -> Result<()> {
        match &self.sock {
            Sock::Classic(s) => {
                if frame.fd {
                    return Err(Error::InvalidFrame(
                        "a CAN FD frame on a bus opened without fd=True".into(),
                    ));
                }
                s.write_frame_insist(&to_classic(frame)?)?;
            }
            Sock::Fd(s) => {
                if frame.fd {
                    s.write_frame_insist(&to_fd(frame)?)?;
                } else {
                    s.write_frame_insist(&to_classic(frame)?)?;
                }
            }
        }
        Ok(())
    }

    fn recv(&self, timeout: Option<Duration>) -> Result<Option<Frame>> {
        let deadline = timeout.map(|t| Instant::now() + t);
        loop {
            if self.closed.load(Ordering::Acquire) {
                return Err(Error::BusClosed);
            }
            let slice = match deadline {
                None => SHUTDOWN_POLL,
                Some(d) => d
                    .saturating_duration_since(Instant::now())
                    .min(SHUTDOWN_POLL),
            };
            if wait_readable(self.raw_fd(), slice)? {
                break;
            }
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return Ok(None);
            }
        }
        let frame = match &self.sock {
            Sock::Classic(s) => s.read_frame().map(from_classic),
            Sock::Fd(s) => s.read_frame().map(from_any),
        };
        match frame {
            Ok(f) => Ok(Some(f)),
            Err(e) if e.kind() == ErrorKind::WouldBlock => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    fn set_filters(&self, filters: &[Filter]) -> Result<()> {
        if filters.is_empty() {
            with_sock!(&self.sock, s => s.set_filter_accept_all())?;
        } else {
            let kernel: Vec<CanFilter> = filters.iter().map(kernel_filter).collect();
            with_sock!(&self.sock, s => s.set_filters(&kernel))?;
        }
        Ok(())
    }

    fn shutdown(&self) {
        self.closed.store(true, Ordering::Release);
    }
}
