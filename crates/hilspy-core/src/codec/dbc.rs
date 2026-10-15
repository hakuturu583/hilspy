//! `dbc:FILE`: CAN frames <-> the messages of a DBC database.
//!
//! Options: `strict` (default true: every signal given, values within
//! `[min|max]`), `scaling` (default true: physical values; false: raw),
//! `decode_choices` (default true: value-table names instead of numbers).
//! Frames whose identifier the database lacks decode as nameless messages.

use super::Codec;
use crate::dbc::{self as db, Database, DecodeOptions, EncodeOptions};
use crate::error::{Error, Result};
use crate::message::{CanHeader, Message, Packet};
use crate::transport::can_frame::fd_padded_len;
use crate::url::CodecSpec;
use crate::value::Value;

pub(crate) struct Dbc {
    db: Database,
    encode: EncodeOptions,
    decode: DecodeOptions,
}

impl Dbc {
    pub(crate) fn open(spec: CodecSpec) -> Result<Self> {
        let path = spec.argument.ok_or_else(|| {
            Error::Config("the dbc codec needs a file: \"dbc:path/to/file.dbc\"".into())
        })?;
        let mut o = spec.options;
        let strict = o.take_bool("strict", true)?;
        let scaling = o.take_bool("scaling", true)?;
        let decode_choices = o.take_bool("decode_choices", true)?;
        o.finish("the dbc codec")?;
        Ok(Self {
            db: Database::from_file(path)?,
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
        Value::Float(f) => db::Value::Float(*f),
        Value::Str(s) => db::Value::Str(s.clone()),
        Value::Null | Value::Map(_) | Value::Array(_) => {
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
        let values = message
            .fields
            .iter()
            .map(|(k, v)| Ok((k.clone(), to_db(k, v)?)))
            .collect::<Result<Vec<_>>>()?;
        let mut data = m.encode(&values, &self.encode)?;
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

    fn decode(&mut self, packet: Packet) -> Result<Vec<Message>> {
        let Some(h) = packet.can else {
            return Ok(vec![Message::raw(packet)]);
        };
        let Ok(m) = self.db.message_by_id(h.id, h.extended) else {
            return Ok(vec![Message::raw(packet)]);
        };
        let fields = m
            .decode(&packet.data, &self.decode)?
            .into_iter()
            .map(|(k, v)| (k, from_db(v)))
            .collect();
        Ok(vec![Message {
            name: Some(m.name.clone()),
            fields,
            packet: Some(packet),
        }])
    }
}
