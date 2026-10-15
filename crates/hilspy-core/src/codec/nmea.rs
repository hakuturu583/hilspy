//! `nmea`: NMEA 0183 sentences (a GNSS receiver's text output) <-> messages.
//!
//! Decoding splits the byte stream into sentences (one decoder per packet
//! source) and parses them with the `nmea` crate: each becomes a message
//! named after its type (`GGA`, `RMC`, `GSV`, ...) with the crate's fields
//! and `talker` (`GP`, `GN`, ...). Sentences the crate does not parse (an
//! unsupported type, a bad checksum) arrive nameless, with their text as
//! data.
//!
//! Encoding writes `GGA`, `RMC`, `VTG`, `HDT` and `ZDA` from the fields
//! decoding reports (a received sentence can be sent on unchanged); the time
//! and date default to the message's time (its channel's virtual clock), the
//! talker to the `talker` option (default `GP`).

use std::collections::HashMap;

use super::Codec;
use crate::error::{Error, Result};
use crate::message::{now_timestamp, Message, Packet};
use crate::url::CodecSpec;
use crate::value::{from_json, Value};

/// Longest line kept while waiting for its end (NMEA allows 82 bytes).
const MAX_LINE: usize = 1024;
const KNOTS_TO_KMH: f64 = 1.852;

pub(crate) struct Nmea {
    talker: String,
    buffers: HashMap<Option<u64>, Vec<u8>>,
}

impl Nmea {
    pub(crate) fn open(spec: CodecSpec) -> Result<Self> {
        if spec.argument.is_some() {
            return Err(Error::Config("the nmea codec takes no argument".into()));
        }
        let mut o = spec.options;
        let talker = o.take_str("talker")?.unwrap_or_else(|| "GP".into());
        if talker.len() != 2 || !talker.chars().all(|c| c.is_ascii_uppercase()) {
            return Err(Error::Config(format!(
                "option talker: {talker:?} is not two capital letters (GP, GN, ...)"
            )));
        }
        o.finish("the nmea codec")?;
        Ok(Self {
            talker,
            buffers: HashMap::new(),
        })
    }
}

/// A sentence's message: parsed, or nameless with the text as data.
pub(crate) fn sentence_message(line: &str, packet: Packet) -> Message {
    let parsed = nmea::parse_str(line)
        .ok()
        .and_then(|r| serde_json::to_value(r).ok());
    if let Some(serde_json::Value::Object(mut m)) = parsed {
        if m.len() == 1 {
            let (name, body) = m.iter_mut().next().expect("one entry");
            let mut fields = match from_json(body.take()) {
                Value::Map(f) => f,
                _ => Vec::new(),
            };
            if line.starts_with('$') && line.len() > 3 {
                fields.push(("talker".into(), Value::Str(line[1..3].to_owned())));
            }
            return Message {
                name: Some(name.clone()),
                fields,
                packet: Some(packet),
            };
        }
    }
    Message::raw(packet)
}

/// Complete sentences from a stream buffer, removing them (and noise).
pub(crate) fn take_sentences(buf: &mut Vec<u8>) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    loop {
        let Some(start) = buf.iter().position(|b| *b == b'$' || *b == b'!') else {
            buf.clear();
            break;
        };
        buf.drain(..start);
        let Some(end) = buf.iter().position(|b| *b == b'\n') else {
            if buf.len() > MAX_LINE {
                buf.clear();
            }
            break;
        };
        let mut line: Vec<u8> = buf.drain(..=end).collect();
        while line.last().is_some_and(|b| *b == b'\n' || *b == b'\r') {
            line.pop();
        }
        out.push(line);
    }
    out
}

/// `$` + body + `*` + checksum + CRLF.
pub(crate) fn sentence(body: &str) -> Vec<u8> {
    let sum = body.bytes().fold(0u8, |a, b| a ^ b);
    format!("${body}*{sum:02X}\r\n").into_bytes()
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

    fn str(&mut self, name: &str) -> Result<Option<String>> {
        match self.take(name) {
            None => Ok(None),
            Some(Value::Str(s)) => Ok(Some(s)),
            Some(v) => Err(Error::encode(format!("field {name}: {v} is not a string"))),
        }
    }

    fn finish(self, sentence: &str) -> Result<()> {
        match self.0.first() {
            None => Ok(()),
            Some((n, _)) => Err(Error::encode(format!("{sentence} has no field {n:?}"))),
        }
    }
}

/// Seconds of the UTC day from `HH:MM:SS(.fff)`.
fn parse_time(s: &str) -> Result<f64> {
    let bad = || Error::encode(format!("{s:?} is not a time (HH:MM:SS)"));
    let parts: Vec<&str> = s.split(':').collect();
    let [h, m, sec] = parts.as_slice() else {
        return Err(bad());
    };
    let h: f64 = h.parse().map_err(|_| bad())?;
    let m: f64 = m.parse().map_err(|_| bad())?;
    let sec: f64 = sec.parse().map_err(|_| bad())?;
    Ok(h * 3600.0 + m * 60.0 + sec)
}

/// `(year, month, day)` from `YYYY-MM-DD`.
fn parse_date(s: &str) -> Result<(i64, u32, u32)> {
    let bad = || Error::encode(format!("{s:?} is not a date (YYYY-MM-DD)"));
    let parts: Vec<&str> = s.split('-').collect();
    let [y, m, d] = parts.as_slice() else {
        return Err(bad());
    };
    Ok((
        y.parse().map_err(|_| bad())?,
        m.parse().map_err(|_| bad())?,
        d.parse().map_err(|_| bad())?,
    ))
}

/// `hhmmss.ss` of seconds of the day.
fn hhmmss(day_seconds: f64) -> String {
    let cs = (day_seconds * 100.0).round() as i64 % 8_640_000;
    format!(
        "{:02}{:02}{:02}.{:02}",
        cs / 360_000,
        cs / 6000 % 60,
        cs / 100 % 60,
        cs % 100
    )
}

/// `ddmm.mmmmm,N` / `dddmm.mmmmm,E` of degrees (empty fields for None).
fn angle(deg: Option<f64>, width: usize, pos: char, neg: char) -> String {
    let Some(deg) = deg else {
        return ",".into();
    };
    // 1e-5 minute units, so rounding carries into the degrees.
    let units = (deg.abs() * 60.0 * 1e5).round() as i64;
    let (d, m) = (units / 6_000_000, units % 6_000_000);
    format!(
        "{d:0width$}{:02}.{:05},{}",
        m / 100_000,
        m % 100_000,
        if deg < 0.0 { neg } else { pos }
    )
}

fn opt(v: Option<f64>, decimals: usize) -> String {
    v.map_or_else(String::new, |x| format!("{x:.decimals$}"))
}

fn fix_quality(v: Option<Value>) -> Result<u8> {
    const NAMES: [&str; 9] = [
        "Invalid",
        "Gps",
        "DGps",
        "Pps",
        "Rtk",
        "FloatRtk",
        "Estimated",
        "Manual",
        "Simulation",
    ];
    match v {
        None => Ok(1),
        Some(Value::Str(s)) => NAMES
            .iter()
            .position(|n| n.eq_ignore_ascii_case(&s))
            .map(|i| i as u8)
            .ok_or_else(|| {
                Error::encode(format!(
                    "field fix_type: {s:?} is not one of {}",
                    NAMES.join(", ")
                ))
            }),
        Some(v) => v
            .as_i64()
            .and_then(|i| u8::try_from(i).ok())
            .filter(|q| *q <= 8)
            .ok_or_else(|| Error::encode(format!("field fix_type: {v} is not a fix type"))),
    }
}

impl Nmea {
    fn body(&self, name: &str, mut f: Fields, time: f64) -> Result<String> {
        let talker = f.str("talker")?.unwrap_or_else(|| self.talker.clone());
        let day_seconds = time.rem_euclid(86_400.0);
        let (y, mo, d, ..) = crate::time::utc(time.floor() as i64);
        let body = match name {
            "GGA" => {
                let t = f.str("fix_time")?.map(|s| parse_time(&s)).transpose()?;
                let q = fix_quality(f.take("fix_type"))?;
                let lat = f.f64("latitude")?;
                let lon = f.f64("longitude")?;
                let sats = f.f64("fix_satellites")?;
                let hdop = f.f64("hdop")?;
                let alt = f.f64("altitude")?;
                let sep = f.f64("geoid_separation")?;
                format!(
                    "{talker}GGA,{},{},{},{q},{},{},{},M,{},M,,",
                    hhmmss(t.unwrap_or(day_seconds)),
                    angle(lat, 2, 'N', 'S'),
                    angle(lon, 3, 'E', 'W'),
                    sats.map_or_else(String::new, |s| format!("{:02}", s as u32)),
                    opt(hdop, 2),
                    opt(alt, 3),
                    opt(sep, 3),
                )
            }
            "RMC" => {
                let t = f.str("fix_time")?.map(|s| parse_time(&s)).transpose()?;
                let date = f.str("fix_date")?.map(|s| parse_date(&s)).transpose()?;
                let (yy, mm, dd) = date.unwrap_or((y, mo.into(), d.into()));
                let (status, mode) = match f.str("status_of_fix")?.as_deref() {
                    None | Some("Autonomous") => ('A', 'A'),
                    Some("Differential") => ('D', 'D'),
                    Some("Invalid") => ('V', 'N'),
                    Some(s) => {
                        return Err(Error::encode(format!(
                            "field status_of_fix: {s:?} is not Autonomous, Differential or Invalid"
                        )))
                    }
                };
                let lat = f.f64("lat")?;
                let lon = f.f64("lon")?;
                let sog = f.f64("speed_over_ground")?;
                let course = f.f64("true_course")?;
                let var = f.f64("magnetic_variation")?;
                // Decoding reports these; they follow from status_of_fix.
                f.take("faa_mode");
                f.take("nav_status");
                format!(
                    "{talker}RMC,{},{status},{},{},{},{},{:02}{:02}{:02},{},{},{mode}",
                    hhmmss(t.unwrap_or(day_seconds)),
                    angle(lat, 2, 'N', 'S'),
                    angle(lon, 3, 'E', 'W'),
                    opt(sog, 2),
                    opt(course, 2),
                    dd,
                    mm,
                    yy.rem_euclid(100),
                    opt(var.map(f64::abs), 1),
                    match var {
                        Some(v) if v < 0.0 => "W",
                        Some(_) => "E",
                        None => "",
                    },
                )
            }
            "VTG" => {
                let course = f.f64("true_course")?;
                let sog = f.f64("speed_over_ground")?;
                format!(
                    "{talker}VTG,{},T,,M,{},N,{},K,A",
                    opt(course, 2),
                    opt(sog, 2),
                    opt(sog.map(|k| k * KNOTS_TO_KMH), 2),
                )
            }
            "HDT" => format!("{talker}HDT,{},T", opt(f.f64("heading")?, 2)),
            "ZDA" => {
                let t = f.str("utc_time")?.map(|s| parse_time(&s)).transpose()?;
                let day = f.f64("day")?.map_or(d.into(), |v| v as u32);
                let month = f.f64("month")?.map_or(mo.into(), |v| v as u32);
                let year = f.f64("year")?.map_or(y, |v| v as i64);
                let zh = f.f64("local_zone_hours")?.unwrap_or(0.0) as i32;
                let zm = f.f64("local_zone_minutes")?.unwrap_or(0.0) as i32;
                format!(
                    "{talker}ZDA,{},{day:02},{month:02},{year:04},{zh:02},{zm:02}",
                    hhmmss(t.unwrap_or(day_seconds))
                )
            }
            other => {
                return Err(Error::NotFound(format!(
                    "the nmea codec cannot encode {other:?} (GGA, RMC, VTG, HDT, ZDA; \\
                     other sentences: send their text as data)"
                )))
            }
        };
        f.finish(name)?;
        Ok(body)
    }
}

impl Codec for Nmea {
    fn encode(&mut self, message: &Message) -> Result<Vec<Packet>> {
        let name = message.name.as_deref().unwrap_or_default();
        let time = message.timestamp().unwrap_or_else(now_timestamp);
        let body = self.body(name, Fields(message.fields.clone()), time)?;
        Ok(vec![Packet::bytes(sentence(&body))])
    }

    fn decode(&mut self, packet: Packet) -> Result<Vec<Message>> {
        let buf = self.buffers.entry(packet.source).or_default();
        buf.extend_from_slice(&packet.data);
        Ok(take_sentences(buf)
            .into_iter()
            .map(|line| {
                let text = String::from_utf8_lossy(&line).into_owned();
                let p = Packet {
                    data: line,
                    can: None,
                    timestamp: packet.timestamp,
                    source: packet.source,
                };
                sentence_message(&text, p)
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn codec() -> Nmea {
        Nmea::open(CodecSpec::parse("nmea", Default::default()).unwrap()).unwrap()
    }

    fn round_trip(c: &mut Nmea, m: Message) -> Message {
        let bytes = c.encode(&m).unwrap().remove(0).data;
        let mut out = c.decode(Packet::bytes(bytes)).unwrap();
        assert_eq!(out.len(), 1);
        out.remove(0)
    }

    fn f(m: &Message, k: &str) -> Value {
        m.field(k).cloned().unwrap_or(Value::Null)
    }

    #[test]
    fn decodes_a_stream_of_sentences() {
        let mut c = codec();
        let stream = b"noise$GPGGA,092750.000,5321.6802,N,00630.3372,W,1,8,1.03,61.7,M,55.2,M,,*76\r\n\
$GPHDT,274.07,T*03\r\n$GPXYZ,1,2*00\r\n$GPRMC,225446,A,4916.45,N,12311.12,W,000.5,054.7,191194,020.3,E*68\r\n";
        let mut out = Vec::new();
        for chunk in stream.chunks(9) {
            out.extend(c.decode(Packet::bytes(chunk.to_vec())).unwrap());
        }
        let names: Vec<_> = out.iter().map(|m| m.name.clone()).collect();
        assert_eq!(
            names,
            [
                Some("GGA".into()),
                Some("HDT".into()),
                None,
                Some("RMC".into())
            ]
        );
        assert_eq!(f(&out[0], "fix_type"), Value::Str("Gps".into()));
        assert_eq!(f(&out[0], "talker"), Value::Str("GP".into()));
        // The crate keeps some values as f32.
        assert!((f(&out[1], "heading").as_f64().unwrap() - 274.07).abs() < 1e-4);
        assert_eq!(out[2].packet.as_ref().unwrap().data, b"$GPXYZ,1,2*00");
    }

    #[test]
    fn encodes_what_the_crate_decodes() {
        let mut c = codec();
        let mut gga = Message::new(
            "GGA",
            vec![
                ("latitude".into(), Value::Float(35.6895)),
                ("longitude".into(), Value::Float(-139.6917)),
                ("fix_type".into(), Value::Str("Rtk".into())),
                ("fix_satellites".into(), Value::Int(12)),
                ("hdop".into(), Value::Float(0.8)),
                ("altitude".into(), Value::Float(40.0)),
                ("geoid_separation".into(), Value::Float(36.7)),
            ],
        );
        // 2024-02-29T12:34:56.25Z
        gga.packet = Some(Packet {
            timestamp: Some(1_709_210_096.25),
            ..Packet::default()
        });
        let bytes = c.encode(&gga).unwrap().remove(0).data;
        assert!(String::from_utf8_lossy(&bytes).starts_with(
            "$GPGGA,123456.25,3541.37000,N,13941.50200,W,4,12,0.80,40.000,M,36.700,M,,*"
        ));
        let back = round_trip(&mut c, gga.clone());
        assert!(matches!(f(&back, "latitude"), Value::Float(x) if (x - 35.6895).abs() < 1e-7));
        assert!(matches!(f(&back, "longitude"), Value::Float(x) if (x + 139.6917).abs() < 1e-7));
        assert_eq!(f(&back, "fix_type"), Value::Str("Rtk".into()));
        assert_eq!(f(&back, "fix_time"), Value::Str("12:34:56.250".into()));
        // A decoded sentence encodes back to the same sentence.
        let again = round_trip(&mut c, back.clone());
        assert_eq!(again.fields, back.fields);

        let mut rmc = Message::new(
            "RMC",
            vec![
                ("lat".into(), Value::Float(-33.0)),
                ("lon".into(), Value::Float(151.0)),
                ("speed_over_ground".into(), Value::Float(10.5)),
                ("true_course".into(), Value::Float(90.0)),
                ("magnetic_variation".into(), Value::Float(-7.5)),
                ("status_of_fix".into(), Value::Str("Differential".into())),
            ],
        );
        rmc.packet = gga.packet.clone();
        let back = round_trip(&mut c, rmc);
        assert_eq!(f(&back, "fix_date"), Value::Str("2024-02-29".into()));
        assert_eq!(f(&back, "status_of_fix"), Value::Str("Differential".into()));
        assert_eq!(f(&back, "magnetic_variation"), Value::Float(-7.5));
        let back = round_trip(&mut c, back.clone());
        assert_eq!(f(&back, "lat"), Value::Float(-33.0));

        for (name, fields) in [
            (
                "VTG",
                vec![("true_course", 54.7), ("speed_over_ground", 5.5)],
            ),
            ("HDT", vec![("heading", 274.07)]),
            ("ZDA", vec![("day", 4.0), ("month", 7.0), ("year", 2002.0)]),
        ] {
            let m = Message::new(
                name,
                fields
                    .iter()
                    .map(|(k, v)| ((*k).to_owned(), Value::Float(*v)))
                    .collect(),
            );
            let back = round_trip(&mut c, m);
            assert_eq!(back.name.as_deref(), Some(name));
            for (k, v) in fields {
                let got = f(&back, k).as_f64().unwrap();
                assert!((got - v).abs() < 1e-4, "{name}.{k}: {got}");
            }
        }
    }

    #[test]
    fn errors() {
        let mut c = codec();
        let bad = |c: &mut Nmea, m: Message| c.encode(&m).unwrap_err().to_string();
        assert!(bad(&mut c, Message::new("GSV", vec![])).contains("cannot encode"));
        assert!(bad(
            &mut c,
            Message::new("HDT", vec![("x".into(), Value::Int(1))])
        )
        .contains("\"x\""));
        assert!(bad(
            &mut c,
            Message::new("GGA", vec![("fix_type".into(), Value::Str("Great".into()))])
        )
        .contains("fix_type"));
        let o = [("talker".to_owned(), Value::Str("gps".into()))]
            .into_iter()
            .collect();
        assert!(Nmea::open(CodecSpec::parse("nmea", o).unwrap()).is_err());
    }
}
