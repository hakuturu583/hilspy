//! `udp://host:port`: datagrams to and from a peer; each datagram is one
//! packet.
//!
//! * `udp://host:port` sends to `host:port` (from `?bind=addr:port`, default
//!   an ephemeral port) and receives what arrives on the local port.
//! * `udp://addr:port?listen` binds `addr:port` and sends to the last peer
//!   that sent to it (or to `?peer=host:port` before then).

use std::net::{SocketAddr, ToSocketAddrs, UdpSocket};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use super::Transport;
use crate::error::{Error, Result};
use crate::message::{now_timestamp, Packet};
use crate::url::Endpoint;

const CLOSE_POLL: Duration = Duration::from_millis(100);

pub(crate) struct Udp {
    socket: UdpSocket,
    peer: Mutex<Option<SocketAddr>>,
    listen: bool,
    closed: AtomicBool,
}

fn resolve(addr: &str) -> Result<SocketAddr> {
    addr.to_socket_addrs()
        .map_err(|e| Error::Config(format!("cannot resolve {addr:?}: {e}")))?
        .next()
        .ok_or_else(|| Error::Config(format!("{addr:?} resolves to no address")))
}

impl Udp {
    pub(crate) fn open(endpoint: Endpoint) -> Result<Self> {
        let mut o = endpoint.options;
        let listen = o.take_bool("listen", false)?;
        let bind = o.take_str("bind")?;
        let peer = o.take_str("peer")?;
        o.finish("udp://")?;
        let address = resolve(&endpoint.address)?;
        let (local, peer) = if listen {
            (address, peer.map(|p| resolve(&p)).transpose()?)
        } else {
            let any = if address.is_ipv4() {
                "0.0.0.0:0"
            } else {
                "[::]:0"
            };
            (resolve(bind.as_deref().unwrap_or(any))?, Some(address))
        };
        let socket = UdpSocket::bind(local)?;
        Ok(Self {
            socket,
            peer: Mutex::new(peer),
            listen,
            closed: AtomicBool::new(false),
        })
    }

    #[cfg(test)]
    pub(crate) fn local_addr(&self) -> SocketAddr {
        self.socket.local_addr().unwrap()
    }
}

impl Transport for Udp {
    fn send(&self, packet: &Packet) -> Result<()> {
        if self.closed.load(Ordering::Acquire) {
            return Err(Error::Closed);
        }
        let peer = (*self.peer.lock().unwrap()).ok_or_else(|| {
            Error::InvalidPacket(
                "no peer to send to yet: a udp listener replies to the last sender (or pass ?peer=)"
                    .into(),
            )
        })?;
        self.socket.send_to(&packet.data, peer)?;
        Ok(())
    }

    fn recv(&self, timeout: Option<Duration>) -> Result<Option<Packet>> {
        let deadline = timeout.map(|t| Instant::now() + t);
        let mut buf = vec![0u8; 65536];
        loop {
            if self.closed.load(Ordering::Acquire) {
                return Err(Error::Closed);
            }
            let slice = match deadline {
                None => CLOSE_POLL,
                Some(d) => d.saturating_duration_since(Instant::now()).min(CLOSE_POLL),
            };
            // A zero read timeout means "block forever".
            self.socket
                .set_read_timeout(Some(slice.max(Duration::from_millis(1))))?;
            match self.socket.recv_from(&mut buf) {
                Ok((n, from)) => {
                    if self.listen {
                        *self.peer.lock().unwrap() = Some(from);
                    }
                    let mut p = Packet::bytes(buf[..n].to_vec());
                    p.timestamp = Some(now_timestamp());
                    return Ok(Some(p));
                }
                Err(e)
                    if matches!(
                        e.kind(),
                        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                    ) => {}
                // Windows reports an ICMP port-unreachable from an earlier send here.
                Err(e) if e.kind() == std::io::ErrorKind::ConnectionReset => {}
                Err(e) => return Err(e.into()),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::url::Endpoint;

    #[test]
    fn round_trip_and_reply_to_last_sender() {
        let server = Udp::open(Endpoint::parse("udp://127.0.0.1:0?listen").unwrap()).unwrap();
        let port = server.local_addr().port();
        assert!(server.send(&Packet::bytes(b"x".to_vec())).is_err());
        let client =
            Udp::open(Endpoint::parse(&format!("udp://127.0.0.1:{port}")).unwrap()).unwrap();
        client.send(&Packet::bytes(b"ping".to_vec())).unwrap();
        let got = server.recv(Some(Duration::from_secs(2))).unwrap().unwrap();
        assert_eq!(got.data, b"ping");
        server.send(&Packet::bytes(b"pong".to_vec())).unwrap();
        let got = client.recv(Some(Duration::from_secs(2))).unwrap().unwrap();
        assert_eq!(got.data, b"pong");
        assert!(client
            .recv(Some(Duration::from_millis(20)))
            .unwrap()
            .is_none());
    }
}
