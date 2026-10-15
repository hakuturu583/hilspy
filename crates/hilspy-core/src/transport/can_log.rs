//! CAN log files for `file://`: candump (`?format=candump`, Linux), Vector
//! BLF (`.blf`) and ASC (`.asc`), through `socketcan::dump` and `blf_asc`.
//!
//! Reading yields every data frame with its identifier and recorded time
//! (`?realtime` replays them spaced as recorded); the channel ends at the
//! end of the log. Writing (`?write`) records every sent CAN packet with its
//! timestamp (or the time of sending). Options: `interface` (the candump
//! device name, default `can0`), `channel` (the BLF/ASC channel, default 1).

use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use blf_asc::{AscReader, AscWriter, BlfReader, BlfWriter};

use super::Transport;
use crate::error::{Error, Result};
use crate::message::{now_timestamp, CanHeader, Packet};

const CLOSE_POLL: Duration = Duration::from_millis(50);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Format {
    Candump,
    Blf,
    Asc,
}

impl Format {
    pub(crate) fn parse(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "candump" | "log" => Some(Format::Candump),
            "blf" => Some(Format::Blf),
            "asc" => Some(Format::Asc),
            _ => None,
        }
    }

    /// The format of a log path's extension (`.log` stays a raw file: the
    /// name is too common to imply candump).
    pub(crate) fn of_path(path: &str) -> Option<Self> {
        let ext = std::path::Path::new(path).extension()?.to_str()?;
        match ext.to_ascii_lowercase().as_str() {
            "blf" => Some(Format::Blf),
            "asc" => Some(Format::Asc),
            _ => None,
        }
    }
}

enum Reader {
    #[cfg(target_os = "linux")]
    Candump(socketcan::dump::Reader<std::io::BufReader<File>>),
    Blf(Box<BlfReader>),
    Asc(Box<AscReader>),
}

enum Writer {
    #[cfg(target_os = "linux")]
    Candump(BufWriter<File>, String),
    Blf(Box<BlfWriter>),
    Asc(Box<AscWriter>),
    Done,
}

enum Mode {
    Read {
        reader: Reader,
        /// A frame read but not yet due (real-time replay), kept across a
        /// receive that timed out before it.
        pending: Option<Packet>,
        /// Wall time of the first frame and its recorded time, to replay in
        /// real time.
        realtime: Option<Option<(Instant, f64)>>,
    },
    Write(Writer),
}

pub(crate) struct CanLog {
    mode: Mutex<Mode>,
    channel: u16,
    closed: AtomicBool,
}

fn blf_err(path: &str) -> impl Fn(blf_asc::BlfError) -> Error + '_ {
    move |e| Error::Io(std::io::Error::other(format!("{path}: {e}")))
}

fn from_blf(m: &blf_asc::Message) -> Option<Packet> {
    if m.is_error_frame || m.is_remote_frame {
        return None;
    }
    Some(Packet {
        data: m.data.to_vec(),
        can: Some(CanHeader {
            id: m.arbitration_id.0,
            extended: m.is_extended_id,
            fd: m.is_fd,
            brs: m.bitrate_switch,
        }),
        lin: None,
        flexray: None,
        timestamp: Some(m.timestamp),
        source: None,
    })
}

fn to_blf(packet: &Packet, channel: u16) -> Result<blf_asc::Message> {
    let h = packet
        .can
        .ok_or_else(|| Error::InvalidPacket("a CAN log records CAN frames (with an id)".into()))?;
    Ok(blf_asc::Message {
        timestamp: packet.timestamp.unwrap_or_else(now_timestamp),
        arbitration_id: h.id.into(),
        is_extended_id: h.extended,
        is_fd: h.fd,
        bitrate_switch: h.brs,
        dlc: crate::transport::can_frame::dlc(packet.data.len()),
        data: packet.data.clone().into(),
        channel,
        ..blf_asc::Message::default()
    })
}

impl CanLog {
    pub(crate) fn open(
        path: &str,
        format: Format,
        write: bool,
        realtime: bool,
        interface: Option<String>,
        channel: Option<u16>,
    ) -> Result<Self> {
        let io =
            |e: std::io::Error| Error::Io(std::io::Error::new(e.kind(), format!("{path}: {e}")));
        if interface.is_some() && format != Format::Candump {
            return Err(Error::Config("option interface is for candump logs".into()));
        }
        if channel.is_some() && format == Format::Candump {
            return Err(Error::Config(
                "option channel is for BLF/ASC logs (candump: interface)".into(),
            ));
        }
        let channel = channel.unwrap_or(1);
        let mode = if write {
            Mode::Write(match format {
                #[cfg(target_os = "linux")]
                Format::Candump => Writer::Candump(
                    BufWriter::new(File::create(path).map_err(io)?),
                    interface.unwrap_or_else(|| "can0".into()),
                ),
                #[cfg(not(target_os = "linux"))]
                Format::Candump => return Err(candump_needs_linux()),
                Format::Blf => {
                    Writer::Blf(Box::new(BlfWriter::create(path).map_err(blf_err(path))?))
                }
                Format::Asc => Writer::Asc(Box::new(
                    AscWriter::create_with_channel(path, channel).map_err(blf_err(path))?,
                )),
            })
        } else {
            Mode::Read {
                reader: match format {
                    #[cfg(target_os = "linux")]
                    Format::Candump => {
                        Reader::Candump(socketcan::dump::Reader::from_file(path).map_err(io)?)
                    }
                    #[cfg(not(target_os = "linux"))]
                    Format::Candump => return Err(candump_needs_linux()),
                    Format::Blf => {
                        Reader::Blf(Box::new(BlfReader::open(path).map_err(blf_err(path))?))
                    }
                    Format::Asc => {
                        Reader::Asc(Box::new(AscReader::open(path).map_err(blf_err(path))?))
                    }
                },
                pending: None,
                realtime: realtime.then_some(None),
            }
        };
        Ok(Self {
            mode: Mutex::new(mode),
            channel,
            closed: AtomicBool::new(false),
        })
    }
}

#[cfg(not(target_os = "linux"))]
fn candump_needs_linux() -> Error {
    Error::Config("candump logs are only available on Linux (BLF and ASC work everywhere)".into())
}

impl Reader {
    /// The next data frame, or `None` at the end of the log.
    fn next(&mut self) -> Result<Option<Packet>> {
        loop {
            let packet = match self {
                #[cfg(target_os = "linux")]
                Reader::Candump(r) => {
                    let Some(rec) = r
                        .next_record()
                        .map_err(|e| Error::decode(format!("candump log: {e}")))?
                    else {
                        return Ok(None);
                    };
                    super::can_socket::from_any(rec.frame).map(|mut p| {
                        p.timestamp = Some(rec.t_us as f64 * 1e-6);
                        p
                    })
                }
                Reader::Blf(r) => match r.next_message() {
                    Ok(None) => return Ok(None),
                    Ok(Some(m)) => from_blf(&m),
                    Err(e) => return Err(Error::decode(format!("BLF: {e}"))),
                },
                Reader::Asc(r) => match r.next_message() {
                    Ok(None) => return Ok(None),
                    Ok(Some(m)) => from_blf(&m),
                    Err(e) => return Err(Error::decode(format!("ASC: {e}"))),
                },
            };
            if packet.is_some() {
                return Ok(packet);
            }
        }
    }
}

impl Writer {
    fn write(&mut self, packet: &Packet, channel: u16) -> Result<()> {
        // The checks every CAN transport makes (identifier range, payload
        // length; FD when the payload needs it), so the log stays consistent.
        let header = crate::transport::can_frame::check(packet, true)?;
        let normalized = Packet {
            can: Some(header),
            ..packet.clone()
        };
        let packet = &normalized;
        match self {
            #[cfg(target_os = "linux")]
            Writer::Candump(w, device) => {
                let frame = super::can_socket::to_any(packet)?;
                let t = packet.timestamp.unwrap_or_else(now_timestamp);
                let rec = socketcan::dump::CanDumpRecord {
                    t_us: (t.max(0.0) * 1e6).round() as u64,
                    device: device.clone(),
                    frame,
                    direction: None,
                };
                writeln!(w, "{rec}")?;
                Ok(())
            }
            Writer::Blf(w) => w
                .on_message_received(&to_blf(packet, channel)?)
                .map_err(|e| Error::Io(std::io::Error::other(e.to_string()))),
            Writer::Asc(w) => w
                .on_message_received(&to_blf(packet, channel)?)
                .map_err(|e| Error::Io(std::io::Error::other(e.to_string()))),
            Writer::Done => Err(Error::Closed),
        }
    }

    fn finish(&mut self) {
        match std::mem::replace(self, Writer::Done) {
            #[cfg(target_os = "linux")]
            Writer::Candump(mut w, _) => {
                let _ = w.flush();
            }
            Writer::Blf(mut w) => {
                let _ = w.finish();
            }
            Writer::Asc(mut w) => {
                let _ = w.finish();
            }
            Writer::Done => {}
        }
    }
}

impl Transport for CanLog {
    fn send(&self, packet: &Packet) -> Result<()> {
        if self.closed.load(Ordering::Acquire) {
            return Err(Error::Closed);
        }
        match &mut *self.mode.lock().unwrap() {
            Mode::Write(w) => w.write(packet, self.channel),
            Mode::Read { .. } => Err(Error::Config(
                "this file:// channel reads; open it with ?write to record".into(),
            )),
        }
    }

    fn recv(&self, timeout: Option<Duration>) -> Result<Option<Packet>> {
        let deadline = timeout.map(|t| Instant::now() + t);
        let mut mode = self.mode.lock().unwrap();
        let Mode::Read {
            reader,
            pending,
            realtime,
        } = &mut *mode
        else {
            drop(mode);
            // A recording receives nothing: wait out the timeout.
            return self.idle(deadline);
        };
        if self.closed.load(Ordering::Acquire) {
            return Err(Error::Closed);
        }
        let packet = match pending.take() {
            Some(p) => p,
            None => match reader.next()? {
                Some(p) => p,
                None => return Err(Error::Closed),
            },
        };
        if let (Some(origin), Some(t)) = (realtime.as_mut(), packet.timestamp) {
            let (start, t0) = *origin.get_or_insert((Instant::now(), t));
            let due = start + Duration::from_secs_f64((t - t0).clamp(0.0, 86_400.0));
            loop {
                if self.closed.load(Ordering::Acquire) {
                    return Err(Error::Closed);
                }
                let now = Instant::now();
                if now >= due {
                    break;
                }
                // Not due before the caller's deadline: keep it for the
                // next receive.
                if deadline.is_some_and(|d| now >= d) {
                    *pending = Some(packet);
                    return Ok(None);
                }
                let wake = deadline.map_or(due, |d| d.min(due));
                std::thread::sleep((wake - now).min(CLOSE_POLL));
            }
        }
        Ok(Some(packet))
    }

    fn close(&self) {
        self.closed.store(true, Ordering::Release);
        if let Mode::Write(w) = &mut *self.mode.lock().unwrap() {
            w.finish();
        }
    }
}

impl CanLog {
    fn idle(&self, deadline: Option<Instant>) -> Result<Option<Packet>> {
        loop {
            if self.closed.load(Ordering::Acquire) {
                return Err(Error::Closed);
            }
            let now = Instant::now();
            if deadline.is_some_and(|d| now >= d) {
                return Ok(None);
            }
            std::thread::sleep(deadline.map_or(CLOSE_POLL, |d| (d - now).min(CLOSE_POLL)));
        }
    }
}

impl Drop for CanLog {
    fn drop(&mut self) {
        if let Ok(Mode::Write(w)) = self.mode.get_mut() {
            w.finish();
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::message::{CanHeader, Packet};
    use crate::transport::open;
    use crate::Error;
    use std::time::{Duration, Instant};

    fn frames() -> Vec<Packet> {
        let p = |id: u32, extended: bool, fd: bool, data: &[u8], t: f64| Packet {
            data: data.to_vec(),
            can: Some(CanHeader {
                id,
                extended,
                fd,
                brs: fd,
            }),
            lin: None,
            flexray: None,
            timestamp: Some(t),
            source: None,
        };
        vec![
            p(0x123, false, false, &[1, 2, 3], 1_700_000_000.0),
            p(0x18DA_F110, true, false, &[0xFF; 8], 1_700_000_000.05),
            p(0x400, false, true, &[7; 64], 1_700_000_000.1),
            p(0x7FF, false, false, &[], 1_700_000_000.15),
        ]
    }

    fn round_trip(name: &str, query: &str) {
        let dir = std::env::temp_dir().join(format!("hilspy-log-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        let w = open(&format!("file://{}?write{query}", path.display())).unwrap();
        for f in frames() {
            w.send(&f).unwrap();
        }
        assert!(w.send(&Packet::bytes(vec![1])).is_err(), "needs an id");
        w.close();
        let r = open(&format!("file://{}?realtime{query}", path.display())).unwrap();
        let start = Instant::now();
        let mut got = Vec::new();
        loop {
            match r.recv(None) {
                Ok(Some(p)) => got.push(p),
                Ok(None) => unreachable!(),
                Err(Error::Closed) => break,
                Err(e) => panic!("{name}: {e}"),
            }
        }
        // Replayed in real time: the last frame 150 ms after the first.
        assert!(start.elapsed() >= Duration::from_millis(140), "{name}");
        assert_eq!(got.len(), 4, "{name}");
        for (g, w) in got.iter().zip(frames()) {
            assert_eq!(g.data, w.data, "{name}");
            assert_eq!(g.can, w.can, "{name}");
        }
        // Times keep their spacing (ASC and BLF store them from the start).
        let dt = got[3].timestamp.unwrap() - got[0].timestamp.unwrap();
        assert!((dt - 0.15).abs() < 1e-5, "{name}: {dt}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn blf_round_trip() {
        round_trip("drive.blf", "");
    }

    #[test]
    fn asc_round_trip() {
        round_trip("drive.asc", "");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn candump_round_trip() {
        round_trip("drive.log", "&format=candump&interface=vcan1");
    }

    #[test]
    fn realtime_replay_honours_receive_timeouts() {
        let dir = std::env::temp_dir().join(format!("hilspy-log-{}-gap", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("gap.asc");
        let w = open(&format!("file://{}?write", path.display())).unwrap();
        let mut f = frames();
        f[1].timestamp = Some(f[0].timestamp.unwrap() + 0.3);
        w.send(&f[0]).unwrap();
        w.send(&f[1]).unwrap();
        w.close();
        let r = open(&format!("file://{}?realtime", path.display())).unwrap();
        assert!(r.recv(Some(Duration::from_millis(50))).unwrap().is_some());
        // The next frame is 300 ms away: a 50 ms receive gives up on time
        // and the frame is not lost.
        let start = Instant::now();
        assert!(r.recv(Some(Duration::from_millis(50))).unwrap().is_none());
        assert!(start.elapsed() < Duration::from_millis(150));
        let p = r.recv(Some(Duration::from_secs(2))).unwrap().unwrap();
        assert_eq!(p.can.unwrap().id, f[1].can.unwrap().id);
        assert!(start.elapsed() >= Duration::from_millis(250));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn writes_normalized_frames() {
        let dir = std::env::temp_dir().join(format!("hilspy-log-{}-norm", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("n.blf");
        let w = open(&format!("file://{}?write", path.display())).unwrap();
        let mut long = Packet::can(0x100, vec![1; 12]);
        long.can.as_mut().unwrap().fd = false;
        w.send(&long).unwrap();
        assert!(
            w.send(&Packet::can(0x800, vec![1])).is_ok(),
            "0x800 is extended"
        );
        let mut bad = Packet::can(0x100, vec![1]);
        bad.can.as_mut().unwrap().id = 0x800;
        assert!(w.send(&bad).is_err(), "11-bit id out of range");
        w.close();
        let r = open(&format!("file://{}", path.display())).unwrap();
        let p = r.recv(None).unwrap().unwrap();
        assert!(p.can.unwrap().fd, "a 12-byte payload is an FD frame");
        assert_eq!(p.data, vec![1; 12]);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn bad_options() {
        let err = |url: &str| open(url).err().unwrap().to_string();
        assert!(err("file:///tmp/x.blf?append").contains("append"));
        assert!(err("file:///tmp/x.blf?format=pcap").contains("pcap"));
        assert!(err("file:///tmp/x.blf?write&interface=can0").contains("interface"));
        assert!(err("file:///nonexistent/x.blf").contains("nonexistent"));
    }
}
