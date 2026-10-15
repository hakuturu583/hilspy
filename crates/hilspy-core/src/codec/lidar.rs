//! `lidar:MODEL`: LiDAR data packets <-> point clouds.
//!
//! `MODEL` names a profile (`pandar_xt32`, `vlp16`, ...; see
//! [`crate::lidar`]). Options:
//!
//! | Option | Default | |
//! | --- | --- | --- |
//! | `profiles` | | a serialized `ProfileSet` file searched before the defaults |
//! | `frame` | `scan` | `scan`: one message per revolution; `packet`: one per packet |
//! | `phase` | `0` | azimuth (degrees) where a scan starts |
//! | `rpm` | profile | rotation speed of encoded scans |
//! | `return_mode` | `strongest` | return mode of encoded packets (single-return) |
//! | `pace` | `true` | send a revolution's packets spread over its duration, as the sensor does |
//!
//! Decoded messages are named `scan`, with arrays `x`, `y`, `z` (metres;
//! x forward, y left, z up), `intensity`, `channel`, `azimuth` (degrees),
//! `distance` (metres), `return` (index into the return mode's returns) and
//! `time` (seconds after `timestamp`), and scalars `timestamp` (the first
//! firing, seconds since the Unix epoch), `model`, `return_mode`, `rpm` and
//! `packets`. A scan ends when the azimuth passes `phase`; the first scan
//! after opening is usually partial.
//!
//! Encoding takes a `scan` with `x`, `y`, `z` (each point goes to the
//! channel of the nearest elevation) or `distance`, `azimuth`, `channel` (as
//! decoded), optional `intensity` and `timestamp` (default: now), and
//! returns the data packets of one revolution, firing column by column, the
//! nearest point per channel and column. A decoded scan can be sent back
//! unchanged.

use std::collections::HashMap;

use super::Codec;
use crate::error::{Error, Result};
use crate::lidar::{self, Cloud, Decoded, Encoder, Model, Points};
use crate::message::{now_timestamp, Message, Packet};
use crate::url::CodecSpec;
use crate::value::{Array, Value};

const SCAN: &str = "scan";

#[derive(Default)]
struct Assembly {
    cloud: Cloud,
    /// First firing, nanoseconds since the Unix epoch.
    start: Option<i64>,
    last: Option<f64>,
    packets: i64,
    rpm: Option<u16>,
    return_mode: Option<String>,
}

pub(crate) struct Lidar {
    model: Model,
    encoder: Encoder,
    per_packet: bool,
    pace: bool,
    phase: f64,
    /// One scan in progress per packet source.
    scans: HashMap<Option<u64>, Assembly>,
}

impl Lidar {
    pub(crate) fn open(spec: CodecSpec) -> Result<Self> {
        let mut o = spec.options;
        let name = spec.argument.filter(|a| !a.is_empty()).ok_or_else(|| {
            Error::Config(format!(
                "the lidar codec needs a model: lidar:MODEL (default models: {})",
                lidar::names().join(", ")
            ))
        })?;
        let extra = o.take_str("profiles")?.map(lidar::load).transpose()?;
        let model = Model::find(&name, extra.as_deref())?;
        let per_packet = match o.take_str("frame")?.as_deref() {
            None | Some("scan") => false,
            Some("packet") => true,
            Some(other) => {
                return Err(Error::Config(format!(
                    "option frame: {other:?} is not scan or packet"
                )))
            }
        };
        let phase = o.take_f64("phase")?.unwrap_or(0.0);
        if !phase.is_finite() {
            return Err(Error::Config("option phase must be finite".into()));
        }
        let rpm = o.take_f64("rpm")?;
        let return_mode = o.take_str("return_mode")?;
        let pace = o.take_bool("pace", true)?;
        let encoder = Encoder::new(&model, rpm, return_mode.as_deref())?;
        o.finish("the lidar codec")?;
        Ok(Self {
            model,
            encoder,
            per_packet,
            pace,
            phase,
            scans: HashMap::new(),
        })
    }

    fn message(&self, a: Assembly, data: Vec<u8>, source: Option<u64>) -> Message {
        let c = a.cloud;
        let start_ns = a.start.unwrap_or_default();
        let start = start_ns as f64 * 1e-9;
        let time = c
            .time
            .iter()
            .map(|t| ((t - start_ns) as f64 * 1e-9) as f32)
            .collect();
        let fields = vec![
            ("timestamp".into(), Value::Float(start)),
            ("model".into(), Value::Str(self.model.name().to_owned())),
            (
                "return_mode".into(),
                a.return_mode.map_or(Value::Null, Value::Str),
            ),
            (
                "rpm".into(),
                a.rpm.map_or(Value::Null, |r| Value::Int(r.into())),
            ),
            ("packets".into(), Value::Int(a.packets)),
            ("x".into(), Value::Array(Array::F32(c.x))),
            ("y".into(), Value::Array(Array::F32(c.y))),
            ("z".into(), Value::Array(Array::F32(c.z))),
            ("intensity".into(), Value::Array(Array::U8(c.intensity))),
            ("channel".into(), Value::Array(Array::U16(c.channel))),
            ("azimuth".into(), Value::Array(Array::F32(c.azimuth))),
            ("distance".into(), Value::Array(Array::F32(c.distance))),
            ("return".into(), Value::Array(Array::U8(c.ret))),
            ("time".into(), Value::Array(Array::F32(time))),
        ];
        Message {
            name: Some(SCAN.into()),
            fields,
            packet: Some(Packet {
                data,
                can: None,
                timestamp: Some(start),
                source,
            }),
        }
    }

    fn decode_scan(&mut self, d: Decoded, source: Option<u64>) -> Vec<Message> {
        let phase = self.phase;
        let mut out = Vec::new();
        let mut st = self.scans.remove(&source).unwrap_or_default();
        let mut rest = d.cloud;
        let mut taken = 0;
        // The packet counts in the scan its first firing belongs to, and in
        // every scan it starts.
        let mut counted = false;
        for (index, azimuth, time) in d.groups {
            let rel = (azimuth - phase).rem_euclid(360.0);
            if st.last.is_some_and(|last| rel < last) {
                let tail = rest.split_off(index - taken);
                st.cloud.append(&mut rest);
                rest = tail;
                taken = index;
                let done = std::mem::take(&mut st);
                out.push(self.message(done, Vec::new(), source));
                st.packets = 1;
                counted = true;
            }
            if !counted {
                st.packets += 1;
                counted = true;
            }
            st.start.get_or_insert(time);
            st.last = Some(rel);
        }
        st.cloud.append(&mut rest);
        st.rpm = d.rpm;
        st.return_mode = Some(d.return_mode.name.clone());
        self.scans.insert(source, st);
        out
    }
}

/// The numbers of a field, None if absent or null.
fn numbers(fields: &mut Vec<(String, Value)>, name: &str) -> Result<Option<Vec<f64>>> {
    let Some(i) = fields.iter().position(|(n, _)| n == name) else {
        return Ok(None);
    };
    match fields.remove(i).1 {
        Value::Null => Ok(None),
        v => v
            .to_f64_vec()
            .map(Some)
            .ok_or_else(|| Error::encode(format!("field {name}: {v} is not numbers"))),
    }
}

impl Codec for Lidar {
    fn encode(&mut self, message: &Message) -> Result<Vec<Packet>> {
        let name = message.name.as_deref().unwrap_or_default();
        if name != SCAN {
            return Err(Error::NotFound(format!(
                "the lidar codec encodes {SCAN:?} messages, not {name:?}"
            )));
        }
        let mut f = message.fields.clone();
        // Fields decoding reports that encoding derives itself.
        f.retain(|(n, _)| {
            !matches!(
                n.as_str(),
                "time" | "return" | "model" | "return_mode" | "rpm" | "packets"
            )
        });
        let timestamp = match numbers(&mut f, "timestamp")?.as_deref() {
            None => message.timestamp().unwrap_or_else(now_timestamp),
            Some([t]) => *t,
            Some(_) => return Err(Error::encode("field timestamp: expected one number")),
        };
        let intensity = numbers(&mut f, "intensity")?;
        let (distance, azimuth, channel) = (
            numbers(&mut f, "distance")?,
            numbers(&mut f, "azimuth")?,
            numbers(&mut f, "channel")?,
        );
        let (x, y, z) = (
            numbers(&mut f, "x")?,
            numbers(&mut f, "y")?,
            numbers(&mut f, "z")?,
        );
        let points = match (distance, azimuth, channel, x, y, z) {
            (Some(distance), Some(azimuth), Some(channel), ..) => Points::Polar {
                distance,
                azimuth,
                channel,
            },
            (None, None, None, Some(x), Some(y), Some(z)) => Points::Cartesian { x, y, z },
            _ => {
                return Err(Error::encode(
                    "a scan needs x, y and z, or distance, azimuth and channel",
                ))
            }
        };
        if let Some((n, _)) = f.first() {
            return Err(Error::encode(format!("a scan has no field {n:?}")));
        }
        self.encoder
            .encode(&self.model, &points, intensity.as_deref(), timestamp)
    }

    fn paced(&self) -> bool {
        self.pace
    }

    fn decode(&mut self, packet: Packet) -> Result<Vec<Message>> {
        let decoded = self.model.decode(&packet.data, packet.timestamp)?;
        if !self.per_packet {
            return Ok(self.decode_scan(decoded, packet.source));
        }
        let a = Assembly {
            start: decoded.groups.first().map(|g| g.2),
            cloud: decoded.cloud,
            last: None,
            packets: 1,
            rpm: decoded.rpm,
            return_mode: Some(decoded.return_mode.name.clone()),
        };
        Ok(vec![self.message(a, packet.data, packet.source)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::Options;

    fn codec(spec: &str, options: &[(&str, Value)]) -> Lidar {
        let o: Options = options
            .iter()
            .map(|(k, v)| ((*k).to_owned(), v.clone()))
            .collect();
        Lidar::open(CodecSpec::parse(spec, o).unwrap()).unwrap()
    }

    fn arr(v: &[f64]) -> Value {
        Value::Array(Array::F64(v.to_vec()))
    }

    /// A ring of points 10 m away at 0 degrees elevation, every 0.5 degrees.
    fn ring() -> Message {
        let (mut x, mut y, mut z) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..720 {
            let a = (f64::from(i) * 0.5).to_radians();
            x.push(10.0 * a.cos());
            y.push(-10.0 * a.sin());
            z.push(0.0);
        }
        Message::new(
            "scan",
            vec![
                ("x".into(), arr(&x)),
                ("y".into(), arr(&y)),
                ("z".into(), arr(&z)),
            ],
        )
    }

    fn field<'a>(m: &'a Message, name: &str) -> &'a Value {
        m.field(name).unwrap()
    }

    #[test]
    fn scans_are_cut_at_the_phase() {
        let mut c = codec("lidar:vlp16", &[]);
        let mut scan = ring();
        let mut packets = Vec::new();
        for rev in 0..3 {
            scan.fields.retain(|(n, _)| n != "timestamp");
            scan.fields.push((
                "timestamp".into(),
                Value::Float(1000.0 + f64::from(rev) * 0.1),
            ));
            packets.extend(c.encode(&scan).unwrap());
        }
        let mut out = Vec::new();
        for p in packets {
            out.extend(c.decode(p).unwrap());
        }
        // Revolutions 1 and 2 end when the next starts; 3 is in progress.
        assert_eq!(out.len(), 2);
        for (i, m) in out.iter().enumerate() {
            assert_eq!(m.name.as_deref(), Some("scan"));
            let Value::Array(Array::F32(x)) = field(m, "x") else {
                panic!()
            };
            // One point per 0.5 degrees lands in its own firing column.
            assert_eq!(x.len(), 720);
            assert_eq!(field(m, "model"), &Value::Str("vlp16".into()));
            assert_eq!(field(m, "return_mode"), &Value::Str("strongest".into()));
            let Value::Float(t) = field(m, "timestamp") else {
                panic!()
            };
            assert!((t - (1000.0 + i as f64 * 0.1)).abs() < 1e-5, "{t}");
            assert_eq!(m.timestamp(), Some(*t));
            let Value::Array(Array::F32(time)) = field(m, "time") else {
                panic!()
            };
            assert!(time.iter().all(|t| (0.0..0.1).contains(t)));
            assert!(matches!(field(m, "packets"), Value::Int(n) if *n >= 75));
        }
    }

    #[test]
    fn packet_frames_and_sending_a_scan_back() {
        let mut c = codec(
            "lidar:pandar_xt32",
            &[("frame", Value::Str("packet".into()))],
        );
        let mut scan = ring();
        scan.fields.push(("intensity".into(), arr(&[42.0; 720])));
        scan.fields.push(("timestamp".into(), Value::Float(5.0)));
        let packets = c.encode(&scan).unwrap();
        let decoded: Vec<Message> = packets
            .iter()
            .flat_map(|p| c.decode(p.clone()).unwrap())
            .collect();
        assert_eq!(decoded.len(), packets.len());
        assert_eq!(decoded[3].packet.as_ref().unwrap().data, packets[3].data);
        let points: usize = decoded
            .iter()
            .map(|m| match field(m, "x") {
                Value::Array(a) => a.len(),
                _ => 0,
            })
            .sum();
        // Channel 15 of the XT32 is the one nearest the horizon.
        assert_eq!(points, 720);
        // A decoded message (with every field) encodes back to its points.
        let mut again = codec("lidar:xt32", &[]);
        let repacked = again.encode(&decoded[3]).unwrap();
        let back = again.model.decode(&repacked[3].data, None).unwrap();
        let orig = c.model.decode(&packets[3].data, None).unwrap();
        assert_eq!(back.cloud.channel, orig.cloud.channel);
        assert_eq!(back.cloud.distance, orig.cloud.distance);
        assert_eq!(back.cloud.intensity, vec![42; orig.cloud.len()]);
    }

    #[test]
    fn bad_specs_and_messages() {
        let open = |spec: &str, o: &[(&str, Value)]| {
            let o: Options = o
                .iter()
                .map(|(k, v)| ((*k).to_owned(), v.clone()))
                .collect();
            Lidar::open(CodecSpec::parse(spec, o).unwrap())
                .err()
                .unwrap()
                .to_string()
        };
        assert!(open("lidar", &[]).contains("vlp16"));
        assert!(open("lidar:nope", &[]).contains("unknown LiDAR"));
        assert!(open("lidar:vlp16", &[("frame", Value::Str("x".into()))]).contains("frame"));
        assert!(open("lidar:vlp16", &[("typo", Value::Int(1))]).contains("typo"));
        assert!(open(
            "lidar:vlp16",
            &[("profiles", Value::Str("/nonexistent".into()))]
        )
        .contains("/nonexistent"));
        let mut c = codec("lidar:vlp16", &[("rpm", Value::Str("1200".into()))]);
        assert!(c.encode(&Message::new("cloud", vec![])).is_err());
        assert!(c.encode(&Message::new("scan", vec![])).is_err());
        let mut m = ring();
        m.fields.push(("colour".into(), Value::Int(1)));
        assert!(c.encode(&m).unwrap_err().to_string().contains("colour"));
        // 1200 rpm: half as many firings per revolution as at 600.
        assert_eq!(c.encode(&ring()).unwrap().len(), 38);
        assert_eq!(codec("lidar:vlp16", &[]).encode(&ring()).unwrap().len(), 75);
    }
}
