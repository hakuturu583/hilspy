//! `ubx`: u-blox UBX frames <-> messages.
//!
//! Decoding frames the byte stream (one decoder per packet source; bytes
//! between frames, such as interleaved NMEA, are skipped), checks the
//! checksum and parses each frame with the `ublox` crate (protocol 23, M8
//! receivers and later): a message named after the packet (`NAV-PVT`,
//! `NAV-ATT`, `MON-VER`, ...) with the crate's fields in its units (degrees,
//! metres, m/s). Frames the crate does not know are named `UBX-<class>-<id>`
//! (hex) with `class` and `msg_id` only.
//!
//! Encoding writes `NAV-PVT` from the fields decoding reports; the date,
//! time and `itow` default to the message's time (its channel's virtual
//! clock), `fix_type` to 3 (3D) and `flags` to 1 (gnssFixOK).

use std::collections::HashMap;

use super::Codec;
use crate::codec::sbf::gps_time;
use crate::error::{Error, Result};
use crate::message::{now_timestamp, Message, Packet};
use crate::url::CodecSpec;
use crate::value::{from_json, Value};

const SYNC: [u8; 2] = [0xB5, 0x62];
/// Longest payload accepted; a longer length is noise, not a frame.
const MAX_PAYLOAD: usize = 8192;

pub(crate) struct Ubx {
    buffers: HashMap<Option<u64>, Vec<u8>>,
}

impl Ubx {
    pub(crate) fn open(spec: CodecSpec) -> Result<Self> {
        if spec.argument.is_some() {
            return Err(Error::Config("the ubx codec takes no argument".into()));
        }
        spec.options.finish("the ubx codec")?;
        Ok(Self {
            buffers: HashMap::new(),
        })
    }
}

/// Fletcher-8 checksum over class, id, length and payload.
fn checksum(bytes: &[u8]) -> [u8; 2] {
    let (mut a, mut b) = (0u8, 0u8);
    for x in bytes {
        a = a.wrapping_add(*x);
        b = b.wrapping_add(a);
    }
    [a, b]
}

pub(crate) fn frame(class: u8, id: u8, payload: &[u8]) -> Vec<u8> {
    let mut f = Vec::with_capacity(payload.len() + 8);
    f.extend_from_slice(&SYNC);
    f.extend_from_slice(&[class, id]);
    f.extend_from_slice(&(payload.len() as u16).to_le_bytes());
    f.extend_from_slice(payload);
    let ck = checksum(&f[2..]);
    f.extend_from_slice(&ck);
    f
}

/// Complete, checksummed frames from a stream buffer, removing them (and
/// what lies between them).
fn take_frames(buf: &mut Vec<u8>) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    loop {
        let Some(start) = buf.windows(2).position(|w| w == SYNC) else {
            // Keep a trailing first sync byte.
            let keep = usize::from(buf.last() == Some(&SYNC[0]));
            buf.drain(..buf.len() - keep);
            break;
        };
        buf.drain(..start);
        if buf.len() < 6 {
            break;
        }
        let len = usize::from(u16::from_le_bytes([buf[4], buf[5]]));
        if len > MAX_PAYLOAD {
            buf.drain(..1);
            continue;
        }
        if buf.len() < len + 8 {
            break;
        }
        if checksum(&buf[2..6 + len]) != [buf[6 + len], buf[7 + len]] {
            buf.drain(..1);
            continue;
        }
        out.push(buf.drain(..len + 8).collect());
    }
    out
}

/// `NavPvt` -> `NAV-PVT`, `NavHpPosLlh` -> `NAV-HPPOSLLH`.
fn ubx_name(variant: &str) -> String {
    let split = variant[1..]
        .find(|c: char| c.is_ascii_uppercase())
        .map_or(variant.len(), |i| i + 1);
    format!(
        "{}-{}",
        variant[..split].to_ascii_uppercase(),
        variant[split..].to_ascii_uppercase()
    )
}

fn frame_message(raw: Vec<u8>, timestamp: Option<f64>, source: Option<u64>) -> Message {
    let (class, id) = (raw[2], raw[3]);
    let mut parser = ublox::Parser::default_proto();
    let mut it = parser.consume_ubx(&raw);
    let parsed = match it.next() {
        Some(Ok(ublox::UbxPacket::Proto23(p))) => {
            let debug = format!("{p:?}");
            let variant: String = debug
                .chars()
                .take_while(char::is_ascii_alphanumeric)
                .collect();
            let fields = match serde_json::to_value(&p).ok().map(from_json) {
                Some(Value::Map(f)) => f,
                _ => Vec::new(),
            };
            (variant != "Unknown" && !variant.is_empty()).then(|| (ubx_name(&variant), fields))
        }
        _ => None,
    };
    drop(it);
    let (name, fields) = parsed.unwrap_or_else(|| {
        (
            format!("UBX-{class:02X}-{id:02X}"),
            vec![
                ("class".into(), Value::Int(class.into())),
                ("msg_id".into(), Value::Int(id.into())),
            ],
        )
    });
    Message {
        name: Some(name),
        fields,
        packet: Some(Packet {
            data: raw,
            can: None,
            timestamp,
            source,
        }),
    }
}

/// Typed access to a message's fields; consumed fields are removed so the
/// rest can be reported as unknown.
struct Fields(Vec<(String, Value)>);

impl Fields {
    fn take(&mut self, name: &str) -> Option<Value> {
        let i = self.0.iter().position(|(n, _)| n == name)?;
        match self.0.remove(i).1 {
            Value::Null => None,
            v => Some(v),
        }
    }

    fn f64(&mut self, name: &str) -> Result<Option<f64>> {
        self.take(name)
            .map(|v| {
                v.as_f64()
                    .ok_or_else(|| Error::encode(format!("field {name}: {v} is not a number")))
            })
            .transpose()
    }

    /// A number scaled to an integer field (`value / scale`).
    fn scaled<T: TryFrom<i64>>(&mut self, name: &str, scale: f64, default: T) -> Result<T> {
        match self.f64(name)? {
            None => Ok(default),
            Some(x) => {
                let raw = (x / scale).round();
                (raw.is_finite() && raw.abs() < 9.0e15)
                    .then(|| T::try_from(raw as i64).ok())
                    .flatten()
                    .ok_or_else(|| Error::encode(format!("field {name}: {x} is out of range")))
            }
        }
    }

    fn finish(self, packet: &str) -> Result<()> {
        match self.0.first() {
            None => Ok(()),
            Some((n, _)) => Err(Error::encode(format!("{packet} has no field {n:?}"))),
        }
    }
}

fn encode_nav_pvt(mut f: Fields, time: f64) -> Result<Vec<u8>> {
    // Decoding reports these; they are fixed or derived.
    for k in ["class", "msg_id", "reserved1", "flags3"] {
        f.take(k);
    }
    let (_, tow) = gps_time(time);
    let secs = time.floor();
    let (y, mo, d, h, mi, s) = crate::time::utc(secs as i64);
    let nano = ((time - secs) * 1e9).round() as i32;
    let mut p = vec![0u8; 92];
    let mut put = |o: usize, b: &[u8]| p[o..o + b.len()].copy_from_slice(b);
    put(0, &f.scaled::<u32>("itow", 1.0, tow)?.to_le_bytes());
    put(4, &f.scaled::<u16>("year", 1.0, y as u16)?.to_le_bytes());
    put(6, &[f.scaled("month", 1.0, mo)?]);
    put(7, &[f.scaled("day", 1.0, d)?]);
    put(8, &[f.scaled("hour", 1.0, h)?]);
    put(9, &[f.scaled("min", 1.0, mi)?]);
    put(10, &[f.scaled("sec", 1.0, s)?]);
    // validDate | validTime | fullyResolved.
    put(11, &[f.scaled("valid", 1.0, 0x07u8)?]);
    put(12, &f.scaled::<u32>("time_accuracy", 1.0, 0)?.to_le_bytes());
    put(16, &f.scaled::<i32>("nanosec", 1.0, nano)?.to_le_bytes());
    put(20, &[f.scaled("fix_type", 1.0, 3u8)?]);
    put(21, &[f.scaled("flags", 1.0, 0x01u8)?]);
    put(22, &[f.scaled("flags2", 1.0, 0u8)?]);
    put(23, &[f.scaled("num_satellites", 1.0, 0u8)?]);
    put(24, &f.scaled::<i32>("lon", 1e-7, 0)?.to_le_bytes());
    put(28, &f.scaled::<i32>("lat", 1e-7, 0)?.to_le_bytes());
    put(32, &f.scaled::<i32>("height", 1e-3, 0)?.to_le_bytes());
    put(36, &f.scaled::<i32>("height_msl", 1e-3, 0)?.to_le_bytes());
    put(40, &f.scaled::<u32>("h_acc", 1e-3, 0)?.to_le_bytes());
    put(44, &f.scaled::<u32>("v_acc", 1e-3, 0)?.to_le_bytes());
    put(48, &f.scaled::<i32>("vel_north", 1e-3, 0)?.to_le_bytes());
    put(52, &f.scaled::<i32>("vel_east", 1e-3, 0)?.to_le_bytes());
    put(56, &f.scaled::<i32>("vel_down", 1e-3, 0)?.to_le_bytes());
    put(60, &f.scaled::<i32>("g_speed", 1e-3, 0)?.to_le_bytes());
    put(64, &f.scaled::<i32>("head_motion", 1e-5, 0)?.to_le_bytes());
    put(68, &f.scaled::<u32>("s_acc", 1e-3, 0)?.to_le_bytes());
    put(72, &f.scaled::<u32>("head_acc", 1e-5, 0)?.to_le_bytes());
    put(76, &f.scaled::<u16>("pdop", 1e-2, 0)?.to_le_bytes());
    put(84, &f.scaled::<i32>("head_vehicle", 1e-5, 0)?.to_le_bytes());
    put(
        88,
        &f.scaled::<i16>("magnetic_declination", 1e-2, 0)?
            .to_le_bytes(),
    );
    put(
        90,
        &f.scaled::<u16>("magnetic_declination_accuracy", 1e-2, 0)?
            .to_le_bytes(),
    );
    f.finish("NAV-PVT")?;
    Ok(frame(0x01, 0x07, &p))
}

impl Codec for Ubx {
    fn encode(&mut self, message: &Message) -> Result<Vec<Packet>> {
        let time = message.timestamp().unwrap_or_else(now_timestamp);
        let f = Fields(message.fields.clone());
        let data = match message.name.as_deref().unwrap_or_default() {
            "NAV-PVT" => encode_nav_pvt(f, time)?,
            other => {
                return Err(Error::NotFound(format!(
                    "the ubx codec cannot encode {other:?} (NAV-PVT; other packets: send \
                     their bytes as data)"
                )))
            }
        };
        Ok(vec![Packet::bytes(data)])
    }

    fn decode(&mut self, packet: Packet) -> Result<Vec<Message>> {
        let buf = self.buffers.entry(packet.source).or_default();
        buf.extend_from_slice(&packet.data);
        Ok(take_frames(buf)
            .into_iter()
            .map(|raw| frame_message(raw, packet.timestamp, packet.source))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn codec() -> Ubx {
        Ubx::open(CodecSpec::parse("ubx", Default::default()).unwrap()).unwrap()
    }

    fn f(m: &Message, k: &str) -> f64 {
        m.field(k).and_then(Value::as_f64).unwrap_or(f64::NAN)
    }

    #[test]
    fn nav_pvt_round_trips_through_the_crate() {
        let mut c = codec();
        let mut m = Message::new(
            "NAV-PVT",
            vec![
                ("lat".into(), Value::Float(35.6895)),
                ("lon".into(), Value::Float(139.6917)),
                ("height".into(), Value::Float(40.125)),
                ("height_msl".into(), Value::Float(3.4)),
                ("vel_north".into(), Value::Float(1.5)),
                ("vel_east".into(), Value::Float(-0.25)),
                ("g_speed".into(), Value::Float(1.52)),
                ("head_motion".into(), Value::Float(350.5)),
                ("num_satellites".into(), Value::Int(18)),
                ("pdop".into(), Value::Float(1.2)),
                ("h_acc".into(), Value::Float(0.02)),
            ],
        );
        // 2024-01-01T00:00:00.5Z: GPS TOW Monday 00:00:18.5.
        m.packet = Some(Packet {
            timestamp: Some(1_704_067_200.5),
            ..Packet::default()
        });
        let bytes = c.encode(&m).unwrap().remove(0).data;
        assert_eq!(bytes.len(), 100);
        // Split, with NMEA noise in between.
        let mut stream = b"$GPGGA,1*00\r\n".to_vec();
        stream.extend_from_slice(&bytes);
        stream.extend_from_slice(b"\xB5\x62\x01");
        let mut out = Vec::new();
        for chunk in stream.chunks(7) {
            out.extend(c.decode(Packet::bytes(chunk.to_vec())).unwrap());
        }
        assert_eq!(out.len(), 1);
        let d = &out[0];
        assert_eq!(d.name.as_deref(), Some("NAV-PVT"));
        assert_eq!(d.packet.as_ref().unwrap().data, bytes);
        assert!((f(d, "lat") - 35.6895).abs() < 1e-7);
        assert!((f(d, "lon") - 139.6917).abs() < 1e-7);
        assert!((f(d, "height") - 40.125).abs() < 1e-9);
        assert!((f(d, "vel_east") + 0.25).abs() < 1e-9);
        assert!((f(d, "head_motion") - 350.5).abs() < 1e-9);
        assert_eq!(f(d, "itow"), 86_418_500.0);
        assert_eq!(
            (f(d, "year"), f(d, "month"), f(d, "day")),
            (2024.0, 1.0, 1.0)
        );
        assert_eq!(
            (f(d, "nanosec"), f(d, "fix_type"), f(d, "num_satellites")),
            (5e8, 3.0, 18.0)
        );
        // A decoded packet encodes back to the same bytes.
        assert_eq!(c.encode(d).unwrap().remove(0).data, bytes);
    }

    #[test]
    fn names_unknown_frames_and_rejects_bad_checksums() {
        let mut c = codec();
        let mut bad = frame(0x01, 0x07, &[0; 92]);
        bad[50] ^= 1;
        let mut stream = bad;
        stream.extend(frame(0x7F, 0x01, &[1, 2, 3]));
        let out = c.decode(Packet::bytes(stream)).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name.as_deref(), Some("UBX-7F-01"));
        assert_eq!(ubx_name("NavHpPosLlh"), "NAV-HPPOSLLH");
        assert_eq!(ubx_name("MonVer"), "MON-VER");
        assert!(c.encode(&Message::new("MON-VER", vec![])).is_err());
        assert!(c
            .encode(&Message::new(
                "NAV-PVT",
                vec![("typo".into(), Value::Int(1))]
            ))
            .is_err());
        assert!(c
            .encode(&Message::new(
                "NAV-PVT",
                vec![("lat".into(), Value::Float(1e9))]
            ))
            .is_err());
    }
}
