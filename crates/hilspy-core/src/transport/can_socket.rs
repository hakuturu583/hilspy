//! `can://ifname`: Linux SocketCAN (`can0`, `vcan0`, ...).
//!
//! Options: `fd` (CAN FD socket), `echo` (receive own frames),
//! `filter=ID[:MASK],...` (kernel filters).

use std::io::ErrorKind;
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use socketcan::id::FdFlags;
use socketcan::{
    CanAnyFrame, CanDataFrame, CanFdFrame, CanFdSocket, CanFilter, CanFrame, CanSocket,
    EmbeddedFrame, ExtendedId, Id, Socket, SocketOptions, StandardId,
};

use super::can_frame::{check, Filter};
use super::Transport;
use crate::error::{Error, Result};
use crate::message::{now_timestamp, CanHeader, Packet};
use crate::url::Endpoint;

/// `CAN_EFF_FLAG`: the identifier-kind bit filters match on.
const EFF_FLAG: u32 = 0x8000_0000;
/// How often a receiver waiting without a timeout checks for `close`.
const CLOSE_POLL: Duration = Duration::from_millis(100);

/// Run `$body` with `$s` bound to whichever socket the transport holds.
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

pub(crate) struct SocketCan {
    sock: Sock,
    fd: bool,
    closed: AtomicBool,
}

impl SocketCan {
    pub(crate) fn open(endpoint: Endpoint) -> Result<Self> {
        let mut o = endpoint.options;
        let fd = o.take_bool("fd", false)?;
        let echo = o.take_bool("echo", false)?;
        let filters = match o.take_str("filter")? {
            Some(spec) => Filter::parse_list(&spec)?,
            None => Vec::new(),
        };
        o.finish("can://")?;
        let ifname = endpoint.address;
        let open_error = |e: std::io::Error| {
            Error::Io(std::io::Error::new(
                e.kind(),
                format!(
                    "cannot open SocketCAN interface {ifname:?}: {e} (for a virtual interface: \
                     `sudo modprobe vcan && sudo ip link add dev {ifname} type vcan && \
                     sudo ip link set up {ifname}`)"
                ),
            ))
        };
        let sock = if fd {
            Sock::Fd(CanFdSocket::open(&ifname).map_err(open_error)?)
        } else {
            Sock::Classic(CanSocket::open(&ifname).map_err(open_error)?)
        };
        with_sock!(&sock, s => s.set_recv_own_msgs(echo))?;
        if !filters.is_empty() {
            let kernel: Vec<CanFilter> = filters.iter().map(kernel_filter).collect();
            with_sock!(&sock, s => s.set_filters(&kernel))?;
        }
        Ok(Self {
            sock,
            fd,
            closed: AtomicBool::new(false),
        })
    }
}

fn kernel_filter(f: &Filter) -> CanFilter {
    let id = if f.extended { f.id | EFF_FLAG } else { f.id };
    CanFilter::new(id, f.mask | EFF_FLAG)
}

fn embedded_id(h: &CanHeader) -> Result<Id> {
    let id = if h.extended {
        ExtendedId::new(h.id).map(Id::Extended)
    } else {
        u16::try_from(h.id)
            .ok()
            .and_then(StandardId::new)
            .map(Id::Standard)
    };
    id.ok_or_else(|| Error::InvalidPacket("identifier out of range".into()))
}

fn header<F: EmbeddedFrame>(f: &F, fd: bool, brs: bool) -> CanHeader {
    let (id, extended) = match f.id() {
        Id::Standard(id) => (u32::from(id.as_raw()), false),
        Id::Extended(id) => (id.as_raw(), true),
    };
    CanHeader {
        id,
        extended,
        fd,
        brs,
    }
}

fn packet<F: EmbeddedFrame>(f: &F, fd: bool, brs: bool) -> Packet {
    Packet {
        data: f.data().to_vec(),
        can: Some(header(f, fd, brs)),
        lin: None,
        flexray: None,
        timestamp: Some(now_timestamp()),
        source: None,
    }
}

/// Data frames as packets; remote and error frames are skipped (`None`).
fn from_classic(frame: CanFrame) -> Option<Packet> {
    match frame {
        CanFrame::Data(f) => Some(packet(&f, false, false)),
        CanFrame::Remote(_) | CanFrame::Error(_) => None,
    }
}

pub(super) fn from_any(frame: CanAnyFrame) -> Option<Packet> {
    match frame {
        CanAnyFrame::Normal(f) => Some(packet(&f, false, false)),
        CanAnyFrame::Fd(f) => Some(packet(&f, true, f.is_brs())),
        CanAnyFrame::Remote(_) | CanAnyFrame::Error(_) => None,
    }
}

/// A CAN packet as a SocketCAN frame (classic or FD).
pub(super) fn to_any(packet: &Packet) -> Result<CanAnyFrame> {
    let h = packet
        .can
        .ok_or_else(|| Error::InvalidPacket("a CAN frame needs an identifier".into()))?;
    let id = embedded_id(&h)?;
    let invalid = || Error::InvalidPacket(format!("not a CAN frame: {} bytes", packet.data.len()));
    if h.fd {
        let flags = if h.brs {
            FdFlags::BRS
        } else {
            FdFlags::empty()
        };
        Ok(CanAnyFrame::Fd(
            CanFdFrame::with_flags(id, &packet.data, flags).ok_or_else(invalid)?,
        ))
    } else {
        Ok(CanAnyFrame::Normal(
            CanDataFrame::new(id, &packet.data).ok_or_else(invalid)?,
        ))
    }
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

impl Transport for SocketCan {
    fn send(&self, packet: &Packet) -> Result<()> {
        if self.closed.load(Ordering::Acquire) {
            return Err(Error::Closed);
        }
        let h = check(packet, self.fd)?;
        let id = embedded_id(&h)?;
        let invalid = || Error::InvalidPacket("SocketCAN rejected the frame".into());
        match &self.sock {
            Sock::Classic(s) => {
                let f = CanDataFrame::new(id, &packet.data).ok_or_else(invalid)?;
                s.write_frame_insist(&CanFrame::Data(f))?;
            }
            Sock::Fd(s) if h.fd => {
                let flags = if h.brs {
                    FdFlags::BRS
                } else {
                    FdFlags::empty()
                };
                let f = CanFdFrame::with_flags(id, &packet.data, flags).ok_or_else(invalid)?;
                s.write_frame_insist(&f)?;
            }
            Sock::Fd(s) => {
                let f = CanDataFrame::new(id, &packet.data).ok_or_else(invalid)?;
                s.write_frame_insist(&CanFrame::Data(f))?;
            }
        }
        Ok(())
    }

    fn recv(&self, timeout: Option<Duration>) -> Result<Option<Packet>> {
        let deadline = timeout.map(|t| Instant::now() + t);
        loop {
            if self.closed.load(Ordering::Acquire) {
                return Err(Error::Closed);
            }
            let slice = match deadline {
                None => CLOSE_POLL,
                Some(d) => d.saturating_duration_since(Instant::now()).min(CLOSE_POLL),
            };
            if wait_readable(with_sock!(&self.sock, s => s.as_raw_fd()), slice)? {
                let frame = match &self.sock {
                    Sock::Classic(s) => s.read_frame().map(from_classic),
                    Sock::Fd(s) => s.read_frame().map(from_any),
                };
                match frame {
                    Ok(Some(p)) => return Ok(Some(p)),
                    Ok(None) => {}
                    Err(e) if e.kind() == ErrorKind::WouldBlock => {}
                    Err(e) => return Err(e.into()),
                }
            }
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return Ok(None);
            }
        }
    }

    fn close(&self) {
        self.closed.store(true, Ordering::Release);
    }
}
