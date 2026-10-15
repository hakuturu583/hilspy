//! `serial://PORT`: a serial port (UART, USB-serial); packets are chunks of
//! the byte stream, as with TCP, so stream codecs (`sbf`, NMEA, ...) frame
//! them.
//!
//! `serial:///dev/ttyUSB0?baud=115200`, `serial://COM3?baud=9600`. Options:
//! `baud` (default 115200), `data_bits` (5-8, default 8), `parity` (`none`,
//! `odd`, `even`), `stop_bits` (1, 2), `flow` (`none`, `software`,
//! `hardware`).

use std::io::{Read, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serialport::{DataBits, FlowControl, Parity, SerialPort, StopBits};

use super::Transport;
use crate::error::{Error, Result};
use crate::message::{now_timestamp, Packet};
use crate::url::Endpoint;

const CLOSE_POLL: Duration = Duration::from_millis(100);

pub(crate) struct Serial {
    reader: Mutex<Box<dyn SerialPort>>,
    writer: Mutex<Box<dyn SerialPort>>,
    closed: AtomicBool,
}

fn choice<T: Copy>(name: &str, value: Option<String>, default: T, map: &[(&str, T)]) -> Result<T> {
    let Some(v) = value else {
        return Ok(default);
    };
    map.iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(&v))
        .map(|(_, t)| *t)
        .ok_or_else(|| {
            let known: Vec<_> = map.iter().map(|(k, _)| *k).collect();
            Error::Config(format!(
                "serial option {name}: {v:?} is not one of {}",
                known.join(", ")
            ))
        })
}

impl Serial {
    pub(crate) fn open(endpoint: Endpoint) -> Result<Self> {
        let mut o = endpoint.options;
        let baud = match o.take_str("baud")? {
            None => 115_200,
            Some(b) => b
                .parse()
                .map_err(|_| Error::Config(format!("serial option baud: {b:?} is not a number")))?,
        };
        let data_bits = choice(
            "data_bits",
            o.take_str("data_bits")?,
            DataBits::Eight,
            &[
                ("5", DataBits::Five),
                ("6", DataBits::Six),
                ("7", DataBits::Seven),
                ("8", DataBits::Eight),
            ],
        )?;
        let parity = choice(
            "parity",
            o.take_str("parity")?,
            Parity::None,
            &[
                ("none", Parity::None),
                ("odd", Parity::Odd),
                ("even", Parity::Even),
            ],
        )?;
        let stop_bits = choice(
            "stop_bits",
            o.take_str("stop_bits")?,
            StopBits::One,
            &[("1", StopBits::One), ("2", StopBits::Two)],
        )?;
        let flow = choice(
            "flow",
            o.take_str("flow")?,
            FlowControl::None,
            &[
                ("none", FlowControl::None),
                ("software", FlowControl::Software),
                ("hardware", FlowControl::Hardware),
            ],
        )?;
        o.finish("serial://")?;
        if endpoint.address.is_empty() {
            return Err(Error::Config(
                "serial:// needs a port: serial:///dev/ttyUSB0 or serial://COM3".into(),
            ));
        }
        let port = serialport::new(&endpoint.address, baud)
            .data_bits(data_bits)
            .parity(parity)
            .stop_bits(stop_bits)
            .flow_control(flow)
            .timeout(CLOSE_POLL)
            .open()
            .map_err(|e| Error::Config(format!("cannot open {}: {e}", endpoint.address)))?;
        let writer = port
            .try_clone()
            .map_err(|e| Error::Config(format!("{}: {e}", endpoint.address)))?;
        Ok(Self {
            reader: Mutex::new(port),
            writer: Mutex::new(writer),
            closed: AtomicBool::new(false),
        })
    }
}

impl Transport for Serial {
    fn send(&self, packet: &Packet) -> Result<()> {
        if self.closed.load(Ordering::Acquire) {
            return Err(Error::Closed);
        }
        let mut w = self.writer.lock().unwrap();
        w.write_all(&packet.data)?;
        w.flush()?;
        Ok(())
    }

    fn recv(&self, timeout: Option<Duration>) -> Result<Option<Packet>> {
        let deadline = timeout.map(|t| Instant::now() + t);
        let mut buf = vec![0u8; 4096];
        let mut port = self.reader.lock().unwrap();
        loop {
            if self.closed.load(Ordering::Acquire) {
                return Err(Error::Closed);
            }
            let slice = match deadline {
                None => CLOSE_POLL,
                Some(d) => d.saturating_duration_since(Instant::now()).min(CLOSE_POLL),
            };
            port.set_timeout(slice.max(Duration::from_millis(1)))
                .map_err(|e| Error::Io(e.into()))?;
            match port.read(&mut buf) {
                Ok(0) => {}
                Ok(n) => {
                    let mut p = Packet::bytes(buf[..n].to_vec());
                    p.timestamp = Some(now_timestamp());
                    return Ok(Some(p));
                }
                Err(e)
                    if matches!(
                        e.kind(),
                        std::io::ErrorKind::TimedOut
                            | std::io::ErrorKind::WouldBlock
                            | std::io::ErrorKind::Interrupted
                    ) => {}
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

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use std::ffi::CStr;
    use std::fs::File;
    use std::os::fd::FromRawFd;

    /// A pseudo-terminal: the master end, and the slave's path.
    fn pty() -> (File, String) {
        unsafe {
            let master = libc::posix_openpt(libc::O_RDWR | libc::O_NOCTTY);
            assert!(master >= 0);
            assert_eq!(libc::grantpt(master), 0);
            assert_eq!(libc::unlockpt(master), 0);
            let name = CStr::from_ptr(libc::ptsname(master))
                .to_str()
                .unwrap()
                .to_owned();
            (File::from_raw_fd(master), name)
        }
    }

    #[test]
    fn round_trip_over_a_pty() {
        let (mut master, path) = pty();
        let port = Serial::open(
            Endpoint::parse(&format!("serial://{path}?baud=9600&parity=none")).unwrap(),
        )
        .unwrap();
        port.send(&Packet::bytes(b"$GPGGA*00\r\n".to_vec()))
            .unwrap();
        let mut got = [0u8; 11];
        master.read_exact(&mut got).unwrap();
        assert_eq!(&got, b"$GPGGA*00\r\n");
        master.write_all(b"pong").unwrap();
        let p = port.recv(Some(Duration::from_secs(2))).unwrap().unwrap();
        assert_eq!(p.data, b"pong");
        assert!(port
            .recv(Some(Duration::from_millis(30)))
            .unwrap()
            .is_none());
        port.close();
        assert!(matches!(port.recv(None), Err(Error::Closed)));
    }

    #[test]
    fn bad_options() {
        let err = |url: &str| {
            Serial::open(Endpoint::parse(url).unwrap())
                .err()
                .unwrap()
                .to_string()
        };
        assert!(err("serial://?baud=9600").contains("needs a port"));
        assert!(err("serial:///dev/null?parity=mark").contains("parity"));
        assert!(err("serial:///dev/null?baud=fast").contains("baud"));
        assert!(err("serial:///dev/null?typo").contains("typo"));
        assert!(err("serial:///nonexistent/tty").contains("cannot open"));
    }
}
