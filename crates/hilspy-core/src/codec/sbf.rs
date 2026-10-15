//! `sbf`: a Septentrio Binary Format byte stream <-> SBF blocks.
//!
//! Decoding reassembles blocks across packets (TCP chunks, serial reads,
//! UDP datagrams), checks their CRC and parses the fields of common
//! navigation blocks; each block is one message named after it
//! (`PVTGeodetic`, ..., or `Block<number>` for one without a parser), with
//! `tow` and `wnc` always present.
//!
//! Encoding takes `PVTGeodetic` and `AttEuler` messages (the fields of
//! [`crate::sbf::encode`]); `tow`/`wnc` default to the current GPS time.

use super::Codec;
use crate::error::{Error, Result};
use crate::message::{now_timestamp, Message, Packet};
use crate::sbf::encode::{AttEuler, PvtGeodetic};
use crate::sbf::{Block, Decoder};
use crate::url::CodecSpec;
use crate::value::Value;

/// GPS epoch (1980-01-06) as a Unix time.
const GPS_EPOCH_UNIX: f64 = 315_964_800.0;
/// GPS - UTC in seconds (since 2017-01-01).
const GPS_UTC_LEAP_SECONDS: f64 = 18.0;
const WEEK_MS: u64 = 7 * 24 * 3600 * 1000;

/// `(wnc, tow_ms)` of a Unix time.
pub(crate) fn gps_time(unix: f64) -> (u16, u32) {
    let ms = ((unix - GPS_EPOCH_UNIX + GPS_UTC_LEAP_SECONDS) * 1000.0)
        .round()
        .max(0.0) as u64;
    ((ms / WEEK_MS) as u16, (ms % WEEK_MS) as u32)
}

pub(crate) struct Sbf {
    decoder: Decoder,
}

impl Sbf {
    pub(crate) fn open(spec: CodecSpec) -> Result<Self> {
        if spec.argument.is_some() {
            return Err(Error::Config("the sbf codec takes no argument".into()));
        }
        spec.options.finish("the sbf codec")?;
        Ok(Self {
            decoder: Decoder::new(),
        })
    }
}

fn message(block: Block, timestamp: Option<f64>) -> Message {
    let name = block
        .name
        .map_or_else(|| format!("Block{}", block.number), str::to_owned);
    let mut fields = block.fields.clone().unwrap_or_else(|| {
        vec![
            (
                "tow".into(),
                block.tow.map_or(Value::Null, |t| Value::Int(t.into())),
            ),
            (
                "wnc".into(),
                block.wnc.map_or(Value::Null, |w| Value::Int(w.into())),
            ),
        ]
    });
    fields.push(("revision".into(), Value::Int(block.revision.into())));
    Message {
        name: Some(name),
        fields,
        packet: Some(Packet {
            data: block.raw,
            can: None,
            timestamp,
        }),
    }
}

/// Typed access to a message's fields; consumed fields are removed so the
/// rest can be reported as unknown.
struct Fields(Vec<(String, Value)>);

impl Fields {
    fn take(&mut self, name: &str) -> Option<Value> {
        let i = self.0.iter().position(|(n, _)| n == name)?;
        Some(self.0.remove(i).1)
    }

    fn f64(&mut self, name: &str) -> Result<Option<f64>> {
        match self.take(name) {
            None | Some(Value::Null) => Ok(None),
            Some(v) => v
                .as_f64()
                .map(Some)
                .ok_or_else(|| Error::encode(format!("field {name}: {v} is not a number"))),
        }
    }

    fn int<T: TryFrom<i64>>(&mut self, name: &str) -> Result<Option<T>> {
        match self.take(name) {
            None | Some(Value::Null) => Ok(None),
            Some(v) => v
                .as_i64()
                .and_then(|i| T::try_from(i).ok())
                .map(Some)
                .ok_or_else(|| Error::encode(format!("field {name}: {v} is out of range"))),
        }
    }

    fn finish(self, block: &str) -> Result<()> {
        match self.0.first() {
            None => Ok(()),
            Some((n, _)) => Err(Error::encode(format!("{block} has no field {n:?}"))),
        }
    }
}

fn encode_pvt_geodetic(mut f: Fields, wnc: u16, tow: u32) -> Result<Vec<u8>> {
    let d = PvtGeodetic::default();
    let block = PvtGeodetic {
        tow: f.int("tow")?.or(Some(tow)),
        wnc: f.int("wnc")?.or(Some(wnc)),
        mode: f.int("mode")?.unwrap_or(d.mode),
        error: f.int("error")?.unwrap_or(d.error),
        latitude: f.f64("latitude")?,
        longitude: f.f64("longitude")?,
        height: f.f64("height")?,
        undulation: f.f64("undulation")?,
        vn: f.f64("vn")?,
        ve: f.f64("ve")?,
        vu: f.f64("vu")?,
        cog: f.f64("cog")?,
        rx_clk_bias: f.f64("rx_clk_bias")?,
        rx_clk_drift: f.f64("rx_clk_drift")?,
        time_system: f.int("time_system")?.or(d.time_system),
        datum: f.int("datum")?.or(d.datum),
        nr_sv: f.int("nr_sv")?,
        wa_corr_info: f.int("wa_corr_info")?.unwrap_or(0),
        reference_id: f.int("reference_id")?,
        mean_corr_age: f.int("mean_corr_age")?,
        signal_info: f.int("signal_info")?.unwrap_or(0),
        alert_flag: f.int("alert_flag")?.unwrap_or(0),
        nr_bases: f.int("nr_bases")?.unwrap_or(0),
        ppp_info: f.int("ppp_info")?.unwrap_or(0),
        latency: f.int("latency")?,
        h_accuracy: f.int("h_accuracy")?,
        v_accuracy: f.int("v_accuracy")?,
        misc: f.int("misc")?.unwrap_or(0),
    };
    f.finish("PVTGeodetic")?;
    block.encode()
}

fn encode_att_euler(mut f: Fields, wnc: u16, tow: u32) -> Result<Vec<u8>> {
    let block = AttEuler {
        tow: f.int("tow")?.or(Some(tow)),
        wnc: f.int("wnc")?.or(Some(wnc)),
        nr_sv: f.int("nr_sv")?,
        error: f.int("error")?.unwrap_or(0),
        mode: f.int("mode")?.unwrap_or(0),
        heading: f.f64("heading")?,
        pitch: f.f64("pitch")?,
        roll: f.f64("roll")?,
        pitch_dot: f.f64("pitch_dot")?,
        roll_dot: f.f64("roll_dot")?,
        heading_dot: f.f64("heading_dot")?,
    };
    f.finish("AttEuler")?;
    block.encode()
}

impl Codec for Sbf {
    fn encode(&mut self, message: &Message) -> Result<Vec<Packet>> {
        let (wnc, tow) = gps_time(now_timestamp());
        let fields = Fields(message.fields.clone());
        let data = match message.name.as_deref().unwrap_or_default() {
            "PVTGeodetic" => encode_pvt_geodetic(fields, wnc, tow)?,
            "AttEuler" => encode_att_euler(fields, wnc, tow)?,
            other => {
                return Err(Error::NotFound(format!(
                    "the sbf codec cannot encode {other:?} (PVTGeodetic, AttEuler; \
                     other blocks: send their bytes as data)"
                )))
            }
        };
        Ok(vec![Packet::bytes(data)])
    }

    fn decode(&mut self, packet: Packet) -> Result<Vec<Message>> {
        let ts = packet.timestamp;
        Ok(self
            .decoder
            .feed(&packet.data)
            .into_iter()
            .map(|b| message(b, ts))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gps_time_of_2024() {
        // 2024-01-01T00:00:00Z: week 2295, Monday 00:00:18 GPS time.
        assert_eq!(gps_time(1_704_067_200.0), (2295, (24 * 3600 + 18) * 1000));
    }

    #[test]
    fn encode_then_decode_in_chunks() {
        let mut c = Sbf::open(CodecSpec::parse("sbf", Default::default()).unwrap()).unwrap();
        let pvt = Message::new(
            "PVTGeodetic",
            vec![
                ("mode".into(), Value::Int(4)),
                ("latitude".into(), Value::Float(0.6)),
                ("tow".into(), Value::Int(1000)),
            ],
        );
        let att = Message::new("AttEuler", vec![("heading".into(), Value::Float(90.0))]);
        let mut bytes = c.encode(&pvt).unwrap().remove(0).data;
        bytes.extend(c.encode(&att).unwrap().remove(0).data);
        let mut out = Vec::new();
        for chunk in bytes.chunks(7) {
            out.extend(c.decode(Packet::bytes(chunk.to_vec())).unwrap());
        }
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].name.as_deref(), Some("PVTGeodetic"));
        assert_eq!(out[0].field("tow"), Some(&Value::Int(1000)));
        assert_eq!(out[0].field("latitude"), Some(&Value::Float(0.6)));
        assert_eq!(out[1].field("heading"), Some(&Value::Float(90.0)));
        // tow defaulted to the current GPS time.
        assert!(matches!(out[1].field("tow"), Some(Value::Int(t)) if *t > 0));
        assert!(c
            .encode(&Message::new(
                "PVTGeodetic",
                vec![("typo".into(), Value::Int(1))]
            ))
            .is_err());
        assert!(c.encode(&Message::new("Nope", vec![])).is_err());
    }
}
