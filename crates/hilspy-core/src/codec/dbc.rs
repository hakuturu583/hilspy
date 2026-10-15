//! `dbc:FILE`: CAN frames <-> the messages of a DBC database.
//!
//! Options: `strict` (default true: every signal given, values within
//! `[min|max]`), `scaling` (default true: physical values; false: raw),
//! `decode_choices` (default true: value-table names instead of numbers),
//! `e2e` (AUTOSAR E2E protection per message, see [`super::e2e`]).
//! Frames whose identifier the database lacks decode as nameless messages.
//! A message's `GenMsgCycleTime` is its [`Codec::cycle_time`].

use std::collections::HashMap;
use std::time::Duration;

use super::e2e::{status_name, Protection};
use super::Codec;
use crate::dbc::{self as db, Database, DecodeOptions, EncodeOptions};
use crate::error::{Error, Result};
use crate::message::{CanHeader, Message, Packet};
use crate::transport::can_frame::fd_padded_len;
use crate::url::CodecSpec;
use crate::value::{Options, Value};

pub(crate) struct Dbc {
    db: Database,
    encode: EncodeOptions,
    decode: DecodeOptions,
    /// E2E protection by message name.
    e2e: HashMap<String, Protection>,
}

/// The signal options `dbc:` and `fibex:` share.
pub(crate) struct SignalOptions {
    strict: bool,
    scaling: bool,
    decode_choices: bool,
    e2e: Option<Value>,
}

impl SignalOptions {
    /// Consume `strict`, `scaling`, `decode_choices` and `e2e`.
    pub(crate) fn take(o: &mut Options) -> Result<Self> {
        Ok(Self {
            strict: o.take_bool("strict", true)?,
            scaling: o.take_bool("scaling", true)?,
            decode_choices: o.take_bool("decode_choices", true)?,
            e2e: o.take("e2e"),
        })
    }
}

impl Dbc {
    pub(crate) fn open(spec: CodecSpec) -> Result<Self> {
        let path = spec.argument.ok_or_else(|| {
            Error::Config("the dbc codec needs a file: \"dbc:path/to/file.dbc\"".into())
        })?;
        let mut o = spec.options;
        let options = SignalOptions::take(&mut o)?;
        o.finish("the dbc codec")?;
        Self::new(Database::from_file(path)?, options)
    }

    /// The codec over an already-built database (a DBC file, or the frames
    /// of a FIBEX channel).
    pub(crate) fn new(db: Database, options: SignalOptions) -> Result<Self> {
        let SignalOptions {
            strict,
            scaling,
            decode_choices,
            e2e,
        } = options;
        let e2e = match e2e {
            None | Some(Value::Null) => HashMap::new(),
            Some(Value::Map(entries)) => entries
                .into_iter()
                .map(|(name, config)| {
                    let size = db.message_by_name(&name)?.size;
                    Ok((name.clone(), Protection::new(&name, config, size)?))
                })
                .collect::<Result<_>>()?,
            Some(v) => {
                return Err(Error::Config(format!(
                    "option e2e: expected a mapping of message names to configurations, not {v}"
                )))
            }
        };
        Ok(Self {
            db,
            e2e,
            encode: EncodeOptions { scaling, strict },
            decode: DecodeOptions {
                decode_choices,
                scaling,
                allow_truncated: false,
            },
        })
    }
}

fn to_db(name: &str, v: &Value) -> Result<db::Value> {
    Ok(match v {
        Value::Bool(b) => db::Value::Int(i64::from(*b)),
        Value::Int(i) => db::Value::Int(*i),
        Value::UInt(u) => db::Value::Float(*u as f64),
        Value::Float(f) => db::Value::Float(*f),
        Value::Str(s) => db::Value::Str(s.clone()),
        Value::Null | Value::Map(_) | Value::Array(_) | Value::List(_) => {
            return Err(Error::encode(format!(
                "signal {name}: {v} is not a signal value"
            )))
        }
    })
}

fn from_db(v: db::Value) -> Value {
    match v {
        db::Value::Int(i) => Value::Int(i),
        db::Value::Float(f) => Value::Float(f),
        db::Value::Str(s) => Value::Str(s),
    }
}

impl Codec for Dbc {
    fn encode(&mut self, message: &Message) -> Result<Vec<Packet>> {
        let name = message.name.as_deref().unwrap_or_default();
        let m = self.db.message_by_name(name)?;
        let mut values = message
            .fields
            .iter()
            .filter(|(k, _)| k != "e2e")
            .map(|(k, v)| Ok((k.clone(), to_db(k, v)?)))
            .collect::<Result<Vec<_>>>()?;
        let protection = self.e2e.get_mut(name);
        if let Some(p) = &protection {
            // The protection writes these; strict mode need not see them.
            for s in &p.signals {
                if !values.iter().any(|(k, _)| k == s) {
                    values.push((s.clone(), db::Value::Int(0)));
                }
            }
        }
        let mut data = m.encode(&values, &self.encode)?;
        if let Some(p) = protection {
            let n = m.size.min(data.len());
            p.protect(&mut data[..n])?;
        }
        if m.fd {
            let len = fd_padded_len(data.len())
                .ok_or_else(|| Error::encode(format!("{} bytes exceeds CAN FD", data.len())))?;
            data.resize(len, 0);
        }
        Ok(vec![Packet {
            data,
            can: Some(CanHeader {
                id: m.id,
                extended: m.extended,
                fd: m.fd,
                brs: m.fd,
            }),
            timestamp: None,
            source: None,
        }])
    }

    fn cycle_time(&self, message: &Message) -> Option<Duration> {
        let name = message.name.as_deref()?;
        self.db.message_by_name(name).ok()?.cycle_time
    }

    fn decode(&mut self, packet: Packet) -> Result<Vec<Message>> {
        let Some(h) = packet.can else {
            return Ok(vec![Message::raw(packet)]);
        };
        let Ok(m) = self.db.message_by_id(h.id, h.extended) else {
            return Ok(vec![Message::raw(packet)]);
        };
        let mut fields: Vec<(String, Value)> = m
            .decode(&packet.data, &self.decode)?
            .into_iter()
            .map(|(k, v)| (k, from_db(v)))
            .collect();
        if let Some(p) = self.e2e.get_mut(&m.name) {
            let n = m.size.min(packet.data.len());
            let status = p.check(&packet.data[..n])?;
            fields.push(("e2e".into(), Value::Str(status_name(status))));
        }
        Ok(vec![Message {
            name: Some(m.name.clone()),
            fields,
            packet: Some(packet),
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e2e_codec() -> Dbc {
        let path = format!("{}/../../tests/data/e2e.dbc", env!("CARGO_MANIFEST_DIR"));
        let map = |kv: &[(&str, Value)]| {
            Value::Map(
                kv.iter()
                    .map(|(k, v)| ((*k).to_owned(), v.clone()))
                    .collect(),
            )
        };
        let names = |n: &[&str]| Value::List(n.iter().map(|s| Value::Str((*s).into())).collect());
        let e2e = map(&[
            (
                "BrakeCmd",
                map(&[
                    ("profile", Value::Int(11)),
                    ("data_id", Value::Int(0x123)),
                    ("signals", names(&["Crc", "Counter", "DataIdNibble"])),
                ]),
            ),
            (
                "SafetyStatus",
                map(&[
                    ("profile", Value::Int(5)),
                    ("data_id", Value::Int(0x4242)),
                    ("signals", names(&["Crc", "Counter"])),
                ]),
            ),
        ]);
        let mut o = Options::new();
        o.insert("e2e", e2e);
        Dbc::open(CodecSpec::parse(&format!("dbc:{path}"), o).unwrap()).unwrap()
    }

    fn brake(pressure: f64) -> Message {
        Message::new(
            "BrakeCmd",
            vec![
                ("Pressure".into(), Value::Float(pressure)),
                ("Request".into(), Value::Int(1)),
            ],
        )
    }

    fn status(m: &Message) -> &str {
        match m.field("e2e") {
            Some(Value::Str(s)) => s,
            _ => "",
        }
    }

    #[test]
    fn e2e_counts_protects_and_checks() {
        let mut tx = e2e_codec();
        let mut rx = e2e_codec();
        let mut frames = Vec::new();
        for i in 0..20 {
            frames.push(tx.encode(&brake(f64::from(i))).unwrap().remove(0));
        }
        let decoded: Vec<Message> = frames
            .iter()
            .map(|f| rx.decode(f.clone()).unwrap().remove(0))
            .collect();
        // Profile 11 counts 0..14.
        let counters: Vec<_> = decoded
            .iter()
            .map(|m| m.field("Counter").cloned())
            .collect();
        assert_eq!(counters[14], Some(Value::Int(14)));
        assert_eq!(counters[15], Some(Value::Int(0)));
        assert!(decoded.iter().all(|m| status(m) == "Ok"), "{decoded:?}");
        assert_eq!(decoded[3].field("Pressure"), Some(&Value::Float(3.0)));
        // Replayed, corrupted and skipped frames are caught.
        assert_eq!(
            status(&rx.decode(frames[19].clone()).unwrap()[0]),
            "Repeated"
        );
        let mut next = tx.encode(&brake(1.0)).unwrap().remove(0);
        next.data[2] ^= 0x01;
        assert_eq!(status(&rx.decode(next).unwrap()[0]), "CrcError");
        let _lost = tx.encode(&brake(1.0)).unwrap();
        let _lost = tx.encode(&brake(1.0)).unwrap();
        let after_gap = tx.encode(&brake(1.0)).unwrap().remove(0);
        assert_eq!(status(&rx.decode(after_gap).unwrap()[0]), "WrongSequence");
        // A frame for another data id fails its CRC.
        let mut other = e2e_codec();
        let mut o2 = Options::new();
        let path = format!("{}/../../tests/data/e2e.dbc", env!("CARGO_MANIFEST_DIR"));
        o2.insert(
            "e2e",
            Value::Map(vec![(
                "BrakeCmd".into(),
                Value::Map(vec![
                    ("profile".into(), Value::Int(11)),
                    ("data_id".into(), Value::Int(0x124)),
                    (
                        "signals".into(),
                        Value::List(vec![
                            Value::Str("Crc".into()),
                            Value::Str("Counter".into()),
                            Value::Str("DataIdNibble".into()),
                        ]),
                    ),
                ]),
            )]),
        );
        let mut wrong = Dbc::open(CodecSpec::parse(&format!("dbc:{path}"), o2).unwrap()).unwrap();
        let f = wrong.encode(&brake(1.0)).unwrap().remove(0);
        assert_ne!(status(&other.decode(f).unwrap()[0]), "Ok");
    }

    #[test]
    fn e2e_profile_5_on_can_fd() {
        let mut tx = e2e_codec();
        let mut rx = e2e_codec();
        for i in 0..300 {
            let m = Message::new(
                "SafetyStatus",
                vec![("Speed".into(), Value::Float(f64::from(i % 100)))],
            );
            let f = tx.encode(&m).unwrap().remove(0);
            assert!(f.can.unwrap().fd);
            let d = rx.decode(f).unwrap().remove(0);
            assert_eq!(status(&d), "Ok");
            assert_eq!(d.field("Counter"), Some(&Value::Int(i64::from(i % 256))));
        }
    }

    #[test]
    fn e2e_config_errors() {
        let path = format!("{}/../../tests/data/e2e.dbc", env!("CARGO_MANIFEST_DIR"));
        let open = |cfg: Value| {
            let mut o = Options::new();
            o.insert("e2e", cfg);
            Dbc::open(CodecSpec::parse(&format!("dbc:{path}"), o).unwrap())
                .err()
                .unwrap()
                .to_string()
        };
        let one = |k: &str, v: Value| {
            Value::Map(vec![(
                "BrakeCmd".into(),
                Value::Map(vec![("profile".into(), Value::Int(11)), (k.into(), v)]),
            )])
        };
        assert!(open(Value::Int(1)).contains("mapping"));
        assert!(open(Value::Map(vec![("Nope".into(), Value::Map(vec![]))])).contains("Nope"));
        assert!(
            open(Value::Map(vec![("BrakeCmd".into(), Value::Map(vec![]))]))
                .contains("needs a profile")
        );
        assert!(open(one("typo", Value::Int(1))).contains("typo"));
        assert!(open(one("mode", Value::Str("half".into()))).contains("mode"));
        assert!(open(one("data_id", Value::Int(-1))).contains("data_id"));
        assert!(open(Value::Map(vec![(
            "BrakeCmd".into(),
            Value::Map(vec![("profile".into(), Value::Int(3))])
        )]))
        .contains("unsupported profile"));
    }
}
