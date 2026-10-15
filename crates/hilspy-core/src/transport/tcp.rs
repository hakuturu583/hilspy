//! `tcp://host:port`: a byte stream; each packet is a chunk as it arrives
//! (codecs that need framing, like `sbf`, reassemble it).
//!
//! * `tcp://host:port` connects as a client.
//! * `tcp://addr:port?listen` accepts clients; a send goes to every
//!   connected client and packets from any of them are received.

use std::io::{Read, Write};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::Transport;
use crate::error::{Error, Result};
use crate::message::{now_timestamp, Packet};
use crate::queue::Queue;
use crate::url::Endpoint;

struct Shared {
    queue: Queue,
    clients: Mutex<Vec<TcpStream>>,
    closed: AtomicBool,
}

pub(crate) struct Tcp {
    shared: Arc<Shared>,
    listen: bool,
    local: SocketAddr,
}

fn resolve(addr: &str) -> Result<SocketAddr> {
    addr.to_socket_addrs()
        .map_err(|e| Error::Config(format!("cannot resolve {addr:?}: {e}")))?
        .next()
        .ok_or_else(|| Error::Config(format!("{addr:?} resolves to no address")))
}

/// Read chunks from one stream into the queue until it ends.
fn spawn_reader(shared: Arc<Shared>, mut stream: TcpStream, server: bool) {
    std::thread::Builder::new()
        .name("hilspy-tcp-read".into())
        .spawn(move || {
            let mut buf = vec![0u8; 65536];
            loop {
                match stream.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        let mut p = Packet::bytes(buf[..n].to_vec());
                        p.timestamp = Some(now_timestamp());
                        shared.queue.push(p);
                    }
                }
            }
            // A client channel ends with its connection; a server keeps
            // accepting others.
            if !server {
                shared.queue.close();
            }
        })
        .expect("spawn a reader thread");
}

impl Tcp {
    pub(crate) fn open(endpoint: Endpoint) -> Result<Self> {
        let mut o = endpoint.options;
        let listen = o.take_bool("listen", false)?;
        o.finish("tcp://")?;
        let shared = Arc::new(Shared {
            queue: Queue::default(),
            clients: Mutex::new(Vec::new()),
            closed: AtomicBool::new(false),
        });
        let address = resolve(&endpoint.address)?;
        let local = if listen {
            let listener = TcpListener::bind(address)?;
            let local = listener.local_addr()?;
            let accept_shared = Arc::clone(&shared);
            std::thread::Builder::new()
                .name("hilspy-tcp-accept".into())
                .spawn(move || {
                    for stream in listener.incoming() {
                        if accept_shared.closed.load(Ordering::Acquire) {
                            break;
                        }
                        let Ok(stream) = stream else { continue };
                        let _ = stream.set_nodelay(true);
                        if let Ok(reader) = stream.try_clone() {
                            accept_shared.clients.lock().unwrap().push(stream);
                            spawn_reader(Arc::clone(&accept_shared), reader, true);
                        }
                    }
                })?;
            local
        } else {
            let stream = TcpStream::connect_timeout(&address, Duration::from_secs(10))?;
            stream.set_nodelay(true)?;
            let local = stream.local_addr()?;
            spawn_reader(Arc::clone(&shared), stream.try_clone()?, false);
            shared.clients.lock().unwrap().push(stream);
            local
        };
        Ok(Self {
            shared,
            listen,
            local,
        })
    }
}

impl Transport for Tcp {
    fn send(&self, packet: &Packet) -> Result<()> {
        if self.shared.closed.load(Ordering::Acquire) {
            return Err(Error::Closed);
        }
        let mut clients = self.shared.clients.lock().unwrap();
        if self.listen {
            // Drop clients whose connection broke.
            clients.retain_mut(|c| c.write_all(&packet.data).is_ok());
            Ok(())
        } else {
            match clients.first_mut() {
                Some(c) => Ok(c.write_all(&packet.data)?),
                None => Err(Error::Closed),
            }
        }
    }

    fn recv(&self, timeout: Option<Duration>) -> Result<Option<Packet>> {
        self.shared.queue.pop(timeout)
    }

    fn close(&self) {
        if self.shared.closed.swap(true, Ordering::AcqRel) {
            return;
        }
        for c in self.shared.clients.lock().unwrap().drain(..) {
            let _ = c.shutdown(Shutdown::Both);
        }
        if self.listen {
            // Wake the accept loop so it sees `closed` and exits. A wildcard
            // address cannot be connected to everywhere (Windows): use loopback.
            let mut wake = self.local;
            if wake.ip().is_unspecified() {
                wake.set_ip(match wake {
                    SocketAddr::V4(_) => std::net::Ipv4Addr::LOCALHOST.into(),
                    SocketAddr::V6(_) => std::net::Ipv6Addr::LOCALHOST.into(),
                });
            }
            let _ = TcpStream::connect_timeout(&wake, Duration::from_millis(200));
        }
        self.shared.queue.close();
    }
}

impl Drop for Tcp {
    fn drop(&mut self) {
        self.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn server_broadcasts_and_receives() {
        let server = Tcp::open(Endpoint::parse("tcp://127.0.0.1:0?listen").unwrap()).unwrap();
        let port = server.local.port();
        let a = Tcp::open(Endpoint::parse(&format!("tcp://127.0.0.1:{port}")).unwrap()).unwrap();
        let b = Tcp::open(Endpoint::parse(&format!("tcp://127.0.0.1:{port}")).unwrap()).unwrap();
        // Wait until both are accepted.
        for _ in 0..200 {
            if server.shared.clients.lock().unwrap().len() == 2 {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        server.send(&Packet::bytes(b"hello".to_vec())).unwrap();
        for c in [&a, &b] {
            let mut got = Vec::new();
            while got.len() < 5 {
                got.extend(c.recv(Some(Duration::from_secs(2))).unwrap().unwrap().data);
            }
            assert_eq!(got, b"hello");
        }
        a.send(&Packet::bytes(b"up".to_vec())).unwrap();
        let mut got = Vec::new();
        while got.len() < 2 {
            got.extend(
                server
                    .recv(Some(Duration::from_secs(2)))
                    .unwrap()
                    .unwrap()
                    .data,
            );
        }
        assert_eq!(got, b"up");
        server.close();
        assert!(matches!(
            a.recv(Some(Duration::from_secs(2))),
            Err(Error::Closed)
        ));
    }

    #[test]
    fn connect_refused() {
        // Bind then drop to find a free port nobody listens on.
        let port = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        assert!(Tcp::open(Endpoint::parse(&format!("tcp://127.0.0.1:{port}")).unwrap()).is_err());
    }
}
