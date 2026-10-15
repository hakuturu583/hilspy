//! `file://path`: a file as a transport, to replay a recording or make one.
//!
//! * `file://capture.sbf` reads the file in chunks; the channel ends at the
//!   end of the file (`?follow` waits for more data instead, like
//!   `tail -f`).
//! * `file://out.sbf?write` (or `?append`) writes every sent packet's bytes;
//!   it receives nothing.
//!
//! Only packet bytes are stored in a raw file. CAN logs keep identifiers and
//! times: `file://drive.blf`, `file://drive.asc`, `file://drive.log?format=candump`
//! (see [`super::can_log`]; options `realtime`, `interface`, `channel`).

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use std::sync::Arc;

use super::can_log::{CanLog, Format};
use super::Transport;
use crate::error::{Error, Result};
use crate::message::{now_timestamp, Packet};
use crate::url::Endpoint;

const CHUNK: usize = 65536;
const FOLLOW_POLL: Duration = Duration::from_millis(50);

enum Mode {
    Read { follow: bool },
    Write,
}

pub(crate) struct FileTransport {
    file: Mutex<File>,
    mode: Mode,
    closed: AtomicBool,
}

/// A raw file, or a CAN log by `?format=` or extension.
pub(crate) fn open(endpoint: Endpoint) -> Result<Arc<dyn Transport>> {
    let mut o = endpoint.options;
    let format = match o.take_str("format")? {
        None => Format::of_path(&endpoint.address),
        Some(f) if f.eq_ignore_ascii_case("raw") => None,
        Some(f) => Some(Format::parse(&f).ok_or_else(|| {
            Error::Config(format!(
                "file:// format {f:?} is not raw, candump, blf or asc"
            ))
        })?),
    };
    let Some(format) = format else {
        return Ok(Arc::new(FileTransport::open(&endpoint.address, o)?));
    };
    let write = o.take_bool("write", false)?;
    if o.take_bool("append", false)? || o.take_bool("follow", false)? {
        return Err(Error::Config(
            "CAN logs are read or written whole: no ?append or ?follow".into(),
        ));
    }
    let realtime = o.take_bool("realtime", false)?;
    let interface = o.take_str("interface")?;
    let channel = o
        .take_str("channel")?
        .map(|c| {
            c.parse::<u16>()
                .map_err(|_| Error::Config(format!("option channel: {c:?} is not a number")))
        })
        .transpose()?;
    o.finish("file://")?;
    Ok(Arc::new(CanLog::open(
        &endpoint.address,
        format,
        write,
        realtime,
        interface,
        channel,
    )?))
}

impl FileTransport {
    fn open(path: &str, mut o: crate::value::Options) -> Result<Self> {
        let write = o.take_bool("write", false)?;
        let append = o.take_bool("append", false)?;
        let follow = o.take_bool("follow", false)?;
        o.finish("file://")?;
        let path = path.to_owned();
        let with_path =
            |e: std::io::Error| Error::Io(std::io::Error::new(e.kind(), format!("{path}: {e}")));
        let (file, mode) = if write || append {
            let f = OpenOptions::new()
                .create(true)
                .write(true)
                .append(append)
                .truncate(!append)
                .open(&path)
                .map_err(with_path)?;
            (f, Mode::Write)
        } else {
            (File::open(&path).map_err(with_path)?, Mode::Read { follow })
        };
        Ok(Self {
            file: Mutex::new(file),
            mode,
            closed: AtomicBool::new(false),
        })
    }
}

impl Transport for FileTransport {
    fn send(&self, packet: &Packet) -> Result<()> {
        if self.closed.load(Ordering::Acquire) {
            return Err(Error::Closed);
        }
        match self.mode {
            Mode::Write => Ok(self.file.lock().unwrap().write_all(&packet.data)?),
            Mode::Read { .. } => Err(Error::Config(
                "this file:// channel reads; open it with ?write or ?append to record".into(),
            )),
        }
    }

    fn recv(&self, timeout: Option<Duration>) -> Result<Option<Packet>> {
        let deadline = timeout.map(|t| Instant::now() + t);
        loop {
            if self.closed.load(Ordering::Acquire) {
                return Err(Error::Closed);
            }
            let follow = match self.mode {
                Mode::Read { follow } => follow,
                // A recording receives nothing: wait out the timeout.
                Mode::Write => true,
            };
            if let Mode::Read { .. } = self.mode {
                let mut buf = vec![0u8; CHUNK];
                let n = self.file.lock().unwrap().read(&mut buf)?;
                if n > 0 {
                    buf.truncate(n);
                    let mut p = Packet::bytes(buf);
                    p.timestamp = Some(now_timestamp());
                    return Ok(Some(p));
                }
                if !follow {
                    return Err(Error::Closed);
                }
            }
            let now = Instant::now();
            if deadline.is_some_and(|d| now >= d) {
                return Ok(None);
            }
            let wait = deadline.map_or(FOLLOW_POLL, |d| (d - now).min(FOLLOW_POLL));
            std::thread::sleep(wait);
        }
    }

    fn close(&self) {
        self.closed.store(true, Ordering::Release);
        let _ = self.file.lock().unwrap().flush();
    }
}

#[cfg(test)]
mod tests {
    use crate::message::Packet;
    use crate::transport::open;
    use crate::Error;
    use std::time::Duration;

    #[test]
    fn records_and_replays() {
        let dir = std::env::temp_dir().join(format!("hilspy-file-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("rec.bin");
        let w = open(&format!("file://{}?write", path.display())).unwrap();
        w.send(&Packet::bytes(b"abc".to_vec())).unwrap();
        w.send(&Packet::bytes(b"def".to_vec())).unwrap();
        assert!(w.recv(Some(Duration::from_millis(10))).unwrap().is_none());
        w.close();
        let r = open(&format!("file://{}", path.display())).unwrap();
        assert_eq!(r.recv(None).unwrap().unwrap().data, b"abcdef");
        assert!(matches!(r.recv(None), Err(Error::Closed)));
        assert!(r.send(&Packet::bytes(vec![1])).is_err());
        assert!(open("file:///nonexistent/hilspy").is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
