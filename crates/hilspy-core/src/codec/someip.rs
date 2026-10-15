//! `someip:FILE`: SOME/IP messages <-> the methods, events and fields of the
//! services of an ASAM FIBEX 4 file.
//!
//! `afibex` parses the services and their datatypes; the payloads are
//! serialized per AUTOSAR PRS_SOMEIP ([`payload`]), and the tests check them
//! against the `asomeip` decoder.
//!
//! A message is named `Service.Method` or `Service.Event` (FIBEX short
//! names); field accessors are `Service.get_Field_field`,
//! `Service.set_Field_field` and `Service.changed_Field_field` (the
//! notifier), as `asomeip` names them. Its fields are the header
//! (`service_id`, `method_id`, `length`, `client_id`, `session_id`,
//! `protocol_version`, `interface_version`, `message_type`, `return_code`)
//! followed by the parameters: the input parameters of requests and
//! notifications, the return parameters of responses. Structures and unions
//! become mappings, arrays lists or typed arrays, enums their synonyms
//! (option `decode_choices`, default true). An `ERROR` message has no
//! parameters but its raw `payload`, if any.
//!
//! Encoding takes the same shape: header fields are optional
//! (`message_type` defaults to `REQUEST` for methods and accessors and to
//! `NOTIFICATION` for events and notifiers, `return_code` to `E_OK`,
//! `client_id` to the option `client_id` (default 0; 0 for notifications,
//! PRS_SOMEIP_00925), `session_id` to a
//! counter per method that starts at 1 and skips 0); `length` is computed.
//! `message_type` and `return_code` take names or numbers.
//!
//! Framing: a datagram may carry several messages, and a stream (TCP) is cut
//! by the length field, with one buffer per packet source. Packets with a
//! source (a TCP connection) are a stream, others whole datagrams, whose
//! incomplete remainder is an error rather than joined to the next datagram;
//! the option `stream` overrides that (`stream=True` for a `file://` replay). Messages of
//! services, methods or interface versions the file lacks, SOME/IP-TP
//! segments and SOME/IP-SD decode as nameless messages carrying their bytes.
//! A message whose payload does not decode is an error; the other messages
//! that came with it are delivered by the next decode.

mod payload;

use std::collections::HashMap;
use std::path::Path;

use afibex::fibex::{FibexData, MethodIdType, Parameter};

use self::payload::{param_name, Inherited, Reader, Serializer};
use super::Codec;
use crate::error::{Error, Result};
use crate::message::{Message, Packet};
use crate::url::CodecSpec;
use crate::value::{Array, Options, Value};

const HEADER_LEN: usize = 16;
/// Longest message accepted (a corrupt length field drops the buffer).
const MAX_MESSAGE: usize = 1 << 24;

const REQUEST: u8 = 0x00;
const NOTIFICATION: u8 = 0x02;
const RESPONSE: u8 = 0x80;
const ERROR: u8 = 0x81;
const TP_FLAG: u8 = 0x20;

const MESSAGE_TYPES: &[(u8, &str)] = &[
    (0x00, "REQUEST"),
    (0x01, "REQUEST_NO_RETURN"),
    (0x02, "NOTIFICATION"),
    (0x80, "RESPONSE"),
    (0x81, "ERROR"),
];

const RETURN_CODES: &[(u8, &str)] = &[
    (0x00, "E_OK"),
    (0x01, "E_NOT_OK"),
    (0x02, "E_UNKNOWN_SERVICE"),
    (0x03, "E_UNKNOWN_METHOD"),
    (0x04, "E_NOT_READY"),
    (0x05, "E_NOT_REACHABLE"),
    (0x06, "E_TIMEOUT"),
    (0x07, "E_WRONG_PROTOCOL_VERSION"),
    (0x08, "E_WRONG_INTERFACE_VERSION"),
    (0x09, "E_MALFORMED_MESSAGE"),
    (0x0A, "E_WRONG_MESSAGE_TYPE"),
    (0x0B, "E_E2E_REPEATED"),
    (0x0C, "E_E2E_WRONG_SEQUENCE"),
    (0x0D, "E_E2E"),
    (0x0E, "E_E2E_NOT_AVAILABLE"),
    (0x0F, "E_E2E_NO_NEW_DATA"),
];

/// Field names the header uses (parameters cannot have them).
const HEADER_FIELDS: &[&str] = &[
    "service_id",
    "method_id",
    "length",
    "client_id",
    "session_id",
    "protocol_version",
    "interface_version",
    "message_type",
    "return_code",
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Method,
    Event,
    Getter,
    Setter,
    Notifier,
}

/// One method, event or field accessor of a service.
struct Op {
    name: String,
    service_id: u16,
    major: u8,
    /// Index into the services of `(service_id, major)`.
    service: usize,
    method_id: u16,
    kind: Kind,
}

pub(crate) struct Someip {
    fd: FibexData,
    ops: Vec<Op>,
    by_name: HashMap<String, usize>,
    by_id: HashMap<(u16, u8, u16), usize>,
    client_id: u16,
    decode_choices: bool,
    /// Whether packets are chunks of a byte stream (TCP, a file) whose
    /// messages may span packets, rather than whole datagrams; `None`
    /// decides by packet: a stream if it has a source (a TCP connection).
    stream: Option<bool>,
    sessions: HashMap<(u16, u16), u16>,
    buffers: HashMap<Option<u64>, Vec<u8>>,
    /// Messages decoded before a payload that failed, for the next decode.
    ready: Vec<Message>,
}

fn take_u16(o: &mut Options, key: &str) -> Result<Option<u16>> {
    let bad = |v: &dyn std::fmt::Display| {
        Error::Config(format!(
            "option {key}: {v} is not a 16-bit unsigned integer"
        ))
    };
    let n = match o.take(key) {
        None => return Ok(None),
        Some(Value::Str(s)) => {
            let t = s.trim();
            let parsed = match t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
                Some(hex) => i64::from_str_radix(hex, 16),
                None => t.parse(),
            };
            parsed.map_err(|_| bad(&format!("{s:?}")))?
        }
        Some(v) => v.as_i64().ok_or_else(|| bad(&v))?,
    };
    u16::try_from(n).map(Some).map_err(|_| bad(&n))
}

impl Someip {
    pub(crate) fn open(spec: CodecSpec) -> Result<Self> {
        let path = spec.argument.ok_or_else(|| {
            Error::Config("the someip codec needs a file: \"someip:path/to/services.xml\"".into())
        })?;
        let mut o = spec.options;
        let client_id = take_u16(&mut o, "client_id")?.unwrap_or(0);
        let decode_choices = o.take_bool("decode_choices", true)?;
        let stream = match o.take("stream") {
            None => None,
            Some(v) => {
                let mut one = Options::new();
                one.insert("stream", v);
                Some(one.take_bool("stream", false)?)
            }
        };
        o.finish("the someip codec")?;
        let mut fd = FibexData::new();
        fd.load_fibex_file(Path::new(&path))
            .map_err(|e| Error::FibexParse(format!("{path}: {e}")))?;
        let mut codec = Self {
            fd,
            ops: Vec::new(),
            by_name: HashMap::new(),
            by_id: HashMap::new(),
            client_id,
            decode_choices,
            stream,
            sessions: HashMap::new(),
            buffers: HashMap::new(),
            ready: Vec::new(),
        };
        codec.index();
        if codec.ops.is_empty() {
            return Err(Error::FibexParse(format!(
                "{path} describes no SOME/IP service method, event or field"
            )));
        }
        Ok(codec)
    }

    /// Name every method, event and field accessor.
    fn index(&mut self) {
        let mut keys: Vec<_> = self
            .fd
            .elements
            .services_map_by_sid_major
            .keys()
            .copied()
            .collect();
        keys.sort_unstable();
        for (service_id, major) in keys {
            let services = &self.fd.elements.services_map_by_sid_major[&(service_id, major)];
            for (index, service) in services.iter().enumerate() {
                let service_name = service.short_name.as_deref().unwrap_or(&service.id);
                let mut mids: Vec<_> = service.methods_by_mid.keys().copied().collect();
                mids.sort_unstable();
                for mid in mids {
                    let (kind, name) = match &service.methods_by_mid[&mid] {
                        MethodIdType::Method(m) => {
                            (Kind::Method, m.short_name.clone().unwrap_or(m.id.clone()))
                        }
                        MethodIdType::Event(m) => {
                            (Kind::Event, m.short_name.clone().unwrap_or(m.id.clone()))
                        }
                        MethodIdType::Getter { field } => {
                            (Kind::Getter, format!("get_{}_field", field_name(field)))
                        }
                        MethodIdType::Setter { field } => {
                            (Kind::Setter, format!("set_{}_field", field_name(field)))
                        }
                        MethodIdType::Notifier { field } => (
                            Kind::Notifier,
                            format!("changed_{}_field", field_name(field)),
                        ),
                    };
                    let mut full = format!("{service_name}.{name}");
                    if self.by_name.contains_key(&full) {
                        // Another version or instance of a service of that name.
                        full = format!("{service_name}@{service_id:#06x}v{major}.{name}");
                    }
                    let i = self.ops.len();
                    self.by_name.insert(full.clone(), i);
                    self.by_id.entry((service_id, major, mid)).or_insert(i);
                    self.ops.push(Op {
                        name: full,
                        service_id,
                        major,
                        service: index,
                        method_id: mid,
                        kind,
                    });
                }
            }
        }
    }

    fn serializer(&self) -> Serializer<'_> {
        Serializer {
            fd: &self.fd,
            decode_choices: self.decode_choices,
        }
    }

    /// The parameters a message of `message_type` carries.
    fn params(&self, op: &Op, message_type: u8) -> Vec<&Parameter> {
        let service =
            &self.fd.elements.services_map_by_sid_major[&(op.service_id, op.major)][op.service];
        let response = message_type == RESPONSE;
        match &service.methods_by_mid[&op.method_id] {
            MethodIdType::Method(m) | MethodIdType::Event(m) => {
                let p = if response {
                    &m.return_params
                } else {
                    &m.input_params
                };
                p.iter().collect()
            }
            MethodIdType::Getter { .. } if !response => Vec::new(),
            MethodIdType::Getter { field }
            | MethodIdType::Setter { field }
            | MethodIdType::Notifier { field } => vec![field.as_ref()],
        }
    }

    fn decode_frame(
        &self,
        frame: &[u8],
        timestamp: Option<f64>,
        source: Option<u64>,
    ) -> Result<Message> {
        let u16_at = |i: usize| u16::from_be_bytes([frame[i], frame[i + 1]]);
        let (service_id, method_id) = (u16_at(0), u16_at(2));
        let (client_id, session_id) = (u16_at(8), u16_at(10));
        let [protocol, interface, message_type, return_code] =
            [frame[12], frame[13], frame[14], frame[15]];
        let packet = Packet {
            data: frame.to_vec(),
            can: None,
            lin: None,
            flexray: None,
            timestamp,
            source,
        };
        let op = match self.by_id.get(&(service_id, interface, method_id)) {
            Some(&i) if message_type & TP_FLAG == 0 => &self.ops[i],
            _ => return Ok(Message::raw(packet)),
        };
        let named = |table: &[(u8, &str)], v: u8| match table.iter().find(|(n, _)| *n == v) {
            Some((_, name)) if self.decode_choices => Value::Str((*name).to_owned()),
            _ => Value::Int(v.into()),
        };
        let mut fields = vec![
            ("service_id".to_owned(), Value::Int(service_id.into())),
            ("method_id".to_owned(), Value::Int(method_id.into())),
            ("length".to_owned(), Value::Int((frame.len() - 8) as i64)),
            ("client_id".to_owned(), Value::Int(client_id.into())),
            ("session_id".to_owned(), Value::Int(session_id.into())),
            ("protocol_version".to_owned(), Value::Int(protocol.into())),
            ("interface_version".to_owned(), Value::Int(interface.into())),
            (
                "message_type".to_owned(),
                named(MESSAGE_TYPES, message_type),
            ),
            ("return_code".to_owned(), named(RETURN_CODES, return_code)),
        ];
        let body = &frame[HEADER_LEN..];
        if message_type == ERROR {
            if !body.is_empty() {
                fields.push(("payload".into(), Value::Array(Array::U8(body.to_vec()))));
            }
        } else {
            let ser = self.serializer();
            let mut r = Reader::new(body);
            for p in self.params(op, message_type) {
                let name = param_name(p);
                let path = format!("{}.{name}", op.name);
                if HEADER_FIELDS.contains(&name.as_str()) {
                    return Err(Error::decode(format!(
                        "{path}: the parameter's name is a SOME/IP header field's"
                    )));
                }
                let v = ser.decode_param(p, Inherited::default(), &mut r, &path)?;
                fields.push((name, v));
            }
            // Trailing bytes (a newer interface's additions) are ignored.
        }
        Ok(Message {
            name: Some(op.name.clone()),
            fields,
            packet: Some(packet),
        })
    }

    fn next_session(&mut self, op: (u16, u16)) -> u16 {
        let s = self.sessions.entry(op).or_insert(0);
        *s = if *s == u16::MAX { 1 } else { *s + 1 };
        *s
    }
}

fn field_name(field: &Parameter) -> String {
    field.short_name.clone().unwrap_or_else(|| field.id.clone())
}

/// The bytes of a sequence value.
fn bytes_of(v: &Value) -> Option<Vec<u8>> {
    let byte = |x: f64| (x.fract() == 0.0 && (0.0..=255.0).contains(&x)).then_some(x as u8);
    match v {
        Value::Array(a) => a.to_f64().into_iter().map(byte).collect(),
        Value::List(l) => l.iter().map(|x| x.as_f64().and_then(byte)).collect(),
        _ => None,
    }
}

/// A header field's value: a number (or a name from `table`) up to `max`.
fn header_value(m: &Message, key: &str, max: u32, table: &[(u8, &str)]) -> Result<Option<u32>> {
    let Some(v) = m.field(key) else {
        return Ok(None);
    };
    let n = match v {
        Value::Str(s) => table
            .iter()
            .find(|(_, name)| name.eq_ignore_ascii_case(s))
            .map(|(n, _)| i64::from(*n)),
        v => v.as_i64(),
    };
    n.and_then(|n| u32::try_from(n).ok())
        .filter(|n| *n <= max)
        .map(Some)
        .ok_or_else(|| {
            let names: Vec<_> = table.iter().map(|(_, n)| *n).collect();
            Error::encode(format!(
                "SOME/IP header field {key}: {v} is not {}",
                if names.is_empty() {
                    format!("a number up to {max}")
                } else {
                    format!("a number up to {max} or one of {}", names.join(", "))
                }
            ))
        })
}

impl Codec for Someip {
    fn encode(&mut self, message: &Message) -> Result<Vec<Packet>> {
        let name = message.name.as_deref().unwrap_or_default();
        let &i = self.by_name.get(name).ok_or_else(|| {
            let mut known: Vec<_> = self.by_name.keys().map(String::as_str).collect();
            known.sort_unstable();
            Error::NotFound(format!(
                "no SOME/IP method, event or field accessor {name:?} (known: {})",
                known.join(", ")
            ))
        })?;
        let op = &self.ops[i];
        let (service_id, method_id, major, kind) = (op.service_id, op.method_id, op.major, op.kind);
        for (key, expected) in [("service_id", service_id), ("method_id", method_id)] {
            if let Some(v) = header_value(message, key, 0xFFFF, &[])? {
                if v != u32::from(expected) {
                    return Err(Error::encode(format!(
                        "{name}: {key} {v:#06x} is not the FIBEX's {expected:#06x}"
                    )));
                }
            }
        }
        let default_type = match kind {
            Kind::Event | Kind::Notifier => NOTIFICATION,
            _ => REQUEST,
        };
        let message_type = header_value(message, "message_type", 0xFF, MESSAGE_TYPES)?
            .map_or(default_type, |v| v as u8);
        if message_type & TP_FLAG != 0 {
            return Err(Error::encode(format!(
                "{name}: SOME/IP-TP (message type {message_type:#04x}) is not supported"
            )));
        }
        let return_code =
            header_value(message, "return_code", 0xFF, RETURN_CODES)?.unwrap_or(0) as u8;
        let protocol = header_value(message, "protocol_version", 0xFF, &[])?.unwrap_or(1) as u8;
        if let Some(v) = header_value(message, "interface_version", 0xFF, &[])? {
            if v != u32::from(major) {
                return Err(Error::encode(format!(
                    "{name}: interface_version {v} is not the FIBEX's major version {major}"
                )));
            }
        }
        // Notifications come from the server: client ID 0x0000
        // (PRS_SOMEIP_00925) unless the message gives one.
        let default_client = if message_type == NOTIFICATION {
            0
        } else {
            self.client_id
        };
        let client_id =
            header_value(message, "client_id", 0xFFFF, &[])?.map_or(default_client, |v| v as u16);
        let session_id = header_value(message, "session_id", 0xFFFF, &[])?.map(|v| v as u16);

        let mut body = Vec::new();
        let mut known: Vec<String> = HEADER_FIELDS.iter().map(|s| (*s).to_owned()).collect();
        if message_type == ERROR {
            known.push("payload".into());
            if let Some(v) = message.field("payload") {
                body = bytes_of(v).ok_or_else(|| {
                    Error::encode(format!(
                        "{name}: payload needs a sequence of bytes, not {v}"
                    ))
                })?;
            }
        } else {
            let ser = self.serializer();
            for p in self.params(op, message_type) {
                let pname = param_name(p);
                let path = format!("{name}.{pname}");
                if HEADER_FIELDS.contains(&pname.as_str()) {
                    return Err(Error::encode(format!(
                        "{path}: the parameter's name is a SOME/IP header field's"
                    )));
                }
                let v = message
                    .field(&pname)
                    .ok_or_else(|| Error::encode(format!("{name}: missing parameter {pname:?}")))?;
                ser.encode_param(p, Inherited::default(), v, &path, &mut body)?;
                known.push(pname);
            }
        }
        if let Some((k, _)) = message.fields.iter().find(|(k, _)| !known.contains(k)) {
            return Err(Error::NotFound(format!(
                "{name} ({}) has no parameter {k:?}",
                MESSAGE_TYPES
                    .iter()
                    .find(|(n, _)| *n == message_type)
                    .map_or("message", |(_, s)| s)
            )));
        }
        let length = u32::try_from(body.len() + 8)
            .map_err(|_| Error::encode(format!("{name}: payload too long")))?;
        let session_id = match session_id {
            Some(s) => s,
            None => self.next_session((service_id, method_id)),
        };
        let mut data = Vec::with_capacity(HEADER_LEN + body.len());
        data.extend_from_slice(&service_id.to_be_bytes());
        data.extend_from_slice(&method_id.to_be_bytes());
        data.extend_from_slice(&length.to_be_bytes());
        data.extend_from_slice(&client_id.to_be_bytes());
        data.extend_from_slice(&session_id.to_be_bytes());
        data.extend_from_slice(&[protocol, major, message_type, return_code]);
        data.extend_from_slice(&body);
        Ok(vec![Packet::bytes(data)])
    }

    fn decode(&mut self, packet: Packet) -> Result<Vec<Message>> {
        let mut out = std::mem::take(&mut self.ready);
        let mut buf = self.buffers.remove(&packet.source).unwrap_or_default();
        buf.extend_from_slice(&packet.data);
        let mut used = 0;
        let mut failed = None;
        while buf.len() - used >= 8 {
            let rest = &buf[used..];
            let length = u32::from_be_bytes([rest[4], rest[5], rest[6], rest[7]]) as usize;
            if !(8..=MAX_MESSAGE).contains(&length) {
                failed = Some(Error::decode(format!(
                    "SOME/IP length field {length} out of range; {} buffered bytes dropped",
                    rest.len()
                )));
                used = buf.len();
                break;
            }
            if rest.len() < 8 + length {
                break;
            }
            let frame = &rest[..8 + length];
            used += 8 + length;
            // A payload that does not decode still has a valid frame
            // boundary: report it, and keep decoding the frames after it.
            match self.decode_frame(frame, packet.timestamp, packet.source) {
                Ok(m) => out.push(m),
                Err(e) => {
                    failed.get_or_insert(e);
                }
            }
        }
        buf.drain(..used);
        if self.stream.unwrap_or(packet.source.is_some()) {
            if !buf.is_empty() {
                self.buffers.insert(packet.source, buf);
            }
        } else if !buf.is_empty() && failed.is_none() {
            // A datagram holds whole messages: a remainder is malformed and
            // must not be glued to the next datagram.
            failed = Some(Error::decode(format!(
                "SOME/IP datagram ends inside a message: {} bytes dropped",
                buf.len()
            )));
        }
        match failed {
            None => Ok(out),
            Some(e) => {
                self.ready = out;
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/someip.xml");

    fn open(options: &[(&str, Value)]) -> Result<Someip> {
        let mut o = Options::new();
        for (k, v) in options {
            o.insert(*k, v.clone());
        }
        Someip::open(CodecSpec::parse(&format!("someip:{FIXTURE}"), o)?)
    }

    fn codec() -> Someip {
        open(&[]).unwrap()
    }

    fn map(kv: &[(&str, Value)]) -> Vec<(String, Value)> {
        kv.iter()
            .map(|(k, v)| ((*k).to_owned(), v.clone()))
            .collect()
    }

    fn s(v: &str) -> Value {
        Value::Str(v.into())
    }

    fn position(lat: f64, lon: f64, alt: f64, quality: &str) -> Value {
        Value::Map(map(&[
            ("latitude", Value::Float(lat)),
            ("longitude", Value::Float(lon)),
            ("altitude", Value::Float(alt)),
            ("quality", s(quality)),
        ]))
    }

    fn waypoint(id: i64, speed: i64) -> Value {
        Value::Map(map(&[("id", Value::Int(id)), ("speed", Value::Int(speed))]))
    }

    fn set_route() -> Message {
        Message::new(
            "VehicleService.SetRoute",
            map(&[
                ("origin", position(35.6895, 139.6917, 40.25, "RTK_FIXED")),
                (
                    "waypoints",
                    Value::List(vec![
                        waypoint(1, -20),
                        waypoint(2, 300),
                        waypoint(0xFFFF, 0),
                    ]),
                ),
                ("name", s("Route 1")),
            ]),
        )
    }

    fn set_route_response() -> Message {
        Message::new(
            "VehicleService.SetRoute",
            map(&[
                ("message_type", s("RESPONSE")),
                ("session_id", Value::Int(7)),
                ("accepted", Value::Int(1)),
                ("eta", Value::Array(Array::U32(vec![10, 20, 4_000_000_000]))),
                ("distances", Value::Array(Array::F32(vec![1.5, -2.25]))),
            ]),
        )
    }

    fn status() -> Message {
        Message::new(
            "VehicleService.Status",
            map(&[
                ("speed", Value::Int(-1234)),
                ("timestamp", Value::Int(1_700_000_000_123)),
                ("position", position(-33.5, 151.25, -3.5, "GNSS")),
            ]),
        )
    }

    fn configure(label: &str) -> Message {
        Message::new(
            "VehicleService.Configure",
            map(&[
                ("message_type", s("REQUEST_NO_RETURN")),
                (
                    "setting",
                    Value::Map(map(&[("threshold", Value::Float(0.75))])),
                ),
                ("label", s(label)),
            ]),
        )
    }

    fn encode(c: &mut Someip, m: &Message) -> Vec<u8> {
        let mut p = c.encode(m).unwrap();
        assert_eq!(p.len(), 1);
        p.remove(0).data
    }

    fn decode_one(c: &mut Someip, data: &[u8]) -> Message {
        let mut out = c.decode(Packet::bytes(data.to_vec())).unwrap();
        assert_eq!(out.len(), 1);
        out.remove(0)
    }

    fn params(m: &Message) -> Vec<(String, Value)> {
        m.fields
            .iter()
            .filter(|(k, _)| !HEADER_FIELDS.contains(&k.as_str()))
            .cloned()
            .collect()
    }

    /// `asomeip`'s decoding of a message: its method name and payload JSON
    /// (64-bit integers, which it writes as `"123n"`, as numbers).
    fn asomeip_decode(fd: &FibexData, frame: &[u8]) -> (String, serde_json::Value) {
        fn numbers(v: serde_json::Value) -> serde_json::Value {
            use serde_json::Value as J;
            match v {
                J::String(s) if s.ends_with('n') && s[..s.len() - 1].parse::<i64>().is_ok() => {
                    J::from(s[..s.len() - 1].parse::<i64>().unwrap())
                }
                J::Array(a) => J::Array(a.into_iter().map(numbers).collect()),
                J::Object(o) => J::Object(o.into_iter().map(|(k, v)| (k, numbers(v))).collect()),
                v => v,
            }
        }
        let text = asomeip::utils::decode_someip_header_and_payload(
            fd,
            0,
            &frame[..HEADER_LEN],
            &frame[HEADER_LEN..],
        )
        .unwrap();
        let start = text.find('{').unwrap();
        let end = text.rfind('[').unwrap();
        let method = text[..start].rsplit('.').next().unwrap().to_owned();
        let json =
            serde_json::from_str(&text[start..end]).unwrap_or_else(|e| panic!("{text}: {e}"));
        (method, numbers(json))
    }

    /// Encode, decode, re-encode, and check the payload against `asomeip`.
    fn round_trip(c: &mut Someip, m: &Message) -> (Vec<u8>, Message) {
        let bytes = encode(c, m);
        let d = decode_one(c, &bytes);
        assert_eq!(d.name, m.name);
        assert_eq!(d.packet.as_ref().unwrap().data, bytes);
        // A decoded message encodes back to identical bytes.
        assert_eq!(encode(c, &d), bytes, "{:?}", m.name);
        let (method, json) = asomeip_decode(&c.fd, &bytes);
        let name = m.name.as_deref().unwrap();
        assert_eq!(method, name.rsplit('.').next().unwrap());
        assert_eq!(json, to_json(&Value::Map(params(&d))), "{name}");
        (bytes, d)
    }

    use crate::value::to_json;

    #[test]
    fn method_request_and_response_round_trip() {
        let mut c = codec();
        let (bytes, d) = round_trip(&mut c, &set_route());
        assert_eq!(&bytes[..4], [0x12, 0x34, 0x00, 0x01]);
        assert_eq!(&bytes[12..16], [1, 1, 0, 0]);
        assert_eq!(d.field("message_type"), Some(&s("REQUEST")));
        assert_eq!(d.field("return_code"), Some(&s("E_OK")));
        assert_eq!(d.field("interface_version"), Some(&Value::Int(1)));
        assert_eq!(
            d.field("origin"),
            Some(&position(35.6895, 139.6917, 40.25, "RTK_FIXED"))
        );
        assert_eq!(d.field("waypoints"), set_route().field("waypoints"));
        assert_eq!(d.field("name"), Some(&s("Route 1")));

        let (bytes, d) = round_trip(&mut c, &set_route_response());
        assert_eq!(bytes[14], 0x80);
        assert_eq!(d.field("session_id"), Some(&Value::Int(7)));
        assert_eq!(d.field("accepted"), Some(&Value::Int(1)));
        assert_eq!(
            d.field("eta"),
            Some(&Value::Array(Array::U32(vec![10, 20, 4_000_000_000])))
        );
        assert_eq!(
            d.field("distances"),
            Some(&Value::Array(Array::F32(vec![1.5, -2.25])))
        );
        // distances has a 16-bit length field: 2 * 4 bytes.
        let tail = &bytes[bytes.len() - 10..];
        assert_eq!(&tail[..2], [0, 8]);
    }

    #[test]
    fn event_union_and_fixed_string_round_trip() {
        let mut c = codec();
        let (bytes, d) = round_trip(&mut c, &status());
        assert_eq!(&bytes[2..4], [0x80, 0x01]);
        assert_eq!(d.field("message_type"), Some(&s("NOTIFICATION")));
        // speed is little endian: -1234 = 0xFB2E.
        assert_eq!(&bytes[16..18], [0x2E, 0xFB]);
        assert_eq!(d.field("speed"), Some(&Value::Int(-1234)));
        assert_eq!(d.field("timestamp"), Some(&Value::Int(1_700_000_000_123)));

        let (bytes, d) = round_trip(&mut c, &configure("ABCDEFG"));
        // Union: length 4, type 2 (threshold), the float; then 8 label bytes.
        assert_eq!(
            &bytes[16..],
            [
                0, 0, 0, 4, 0, 0, 0, 2, 0x3F, 0x40, 0, 0, b'A', b'B', b'C', b'D', b'E', b'F', b'G',
                0
            ]
        );
        assert_eq!(
            d.field("setting"),
            Some(&Value::Map(map(&[("threshold", Value::Float(0.75))])))
        );
        // A short label is zero padded and decodes without the padding.
        let short = encode(&mut c, &configure("AB"));
        assert_eq!(&short[short.len() - 8..], [b'A', b'B', 0, 0, 0, 0, 0, 0]);
        let d = decode_one(&mut c, &short);
        assert_eq!(d.field("label"), Some(&s("AB")));
        assert_eq!(encode(&mut c, &d), short);
        let e = c.encode(&configure("ABCDEFGH")).unwrap_err().to_string();
        assert!(e.contains("more than the fixed 8"), "{e}");
    }

    #[test]
    fn field_accessors() {
        let mut c = codec();
        let get = Message::new("VehicleService.get_Mode_field", vec![]);
        let bytes = encode(&mut c, &get);
        assert_eq!(&bytes[..8], [0x12, 0x34, 0, 16, 0, 0, 0, 8]);
        let d = decode_one(&mut c, &bytes);
        assert_eq!(params(&d), vec![]);
        let reply = Message::new(
            "VehicleService.get_Mode_field",
            map(&[("message_type", s("RESPONSE")), ("Mode", s("GNSS"))]),
        );
        let (bytes, d) = round_trip(&mut c, &reply);
        assert_eq!(bytes[16..], [1]);
        assert_eq!(d.field("Mode"), Some(&s("GNSS")));
        let set = Message::new(
            "VehicleService.set_Mode_field",
            map(&[("Mode", Value::Int(2))]),
        );
        let (_, d) = round_trip(&mut c, &set);
        assert_eq!(d.field("Mode"), Some(&s("RTK_FIXED")));
        let changed = Message::new(
            "VehicleService.changed_Mode_field",
            map(&[("Mode", s("NONE"))]),
        );
        let (bytes, _) = round_trip(&mut c, &changed);
        assert_eq!((&bytes[2..4], bytes[14]), (&[0x80, 0x02][..], NOTIFICATION));
    }

    #[test]
    fn decodes_hand_built_bytes() {
        #[rustfmt::skip]
        let frame: Vec<u8> = vec![
            0x12, 0x34, 0x00, 0x01,             // service 0x1234, method 1
            0x00, 0x00, 0x00, 0x33,             // length 8 + 43
            0xAB, 0xCD, 0x00, 0x05,             // client 0xABCD, session 5
            0x01, 0x01, 0x00, 0x00,             // protocol 1, interface 1, REQUEST, E_OK
            // origin: latitude 1.5, longitude -2.0, altitude 0.5, RTK_FIXED
            0x3F, 0xF8, 0, 0, 0, 0, 0, 0,
            0xC0, 0x00, 0, 0, 0, 0, 0, 0,
            0x3F, 0x00, 0, 0,
            0x02,
            // waypoints: 8 bytes, {1, -1}, {0x0203, 300}
            0, 0, 0, 8,
            0x00, 0x01, 0xFF, 0xFF,
            0x02, 0x03, 0x01, 0x2C,
            // name: 6 bytes, BOM "ab" NUL
            0, 0, 0, 6,
            0xEF, 0xBB, 0xBF, b'a', b'b', 0x00,
        ];
        let mut c = codec();
        let d = decode_one(&mut c, &frame);
        assert_eq!(d.name.as_deref(), Some("VehicleService.SetRoute"));
        let expected = map(&[
            ("service_id", Value::Int(0x1234)),
            ("method_id", Value::Int(1)),
            ("length", Value::Int(51)),
            ("client_id", Value::Int(0xABCD)),
            ("session_id", Value::Int(5)),
            ("protocol_version", Value::Int(1)),
            ("interface_version", Value::Int(1)),
            ("message_type", s("REQUEST")),
            ("return_code", s("E_OK")),
            ("origin", position(1.5, -2.0, 0.5, "RTK_FIXED")),
            (
                "waypoints",
                Value::List(vec![waypoint(1, -1), waypoint(0x0203, 300)]),
            ),
            ("name", s("ab")),
        ]);
        assert_eq!(d.fields, expected);
        assert_eq!(encode(&mut c, &d), frame);
        let (_, json) = asomeip_decode(&c.fd, &frame);
        assert_eq!(json, to_json(&Value::Map(params(&d))));

        // Numbers instead of names.
        let mut raw = open(&[("decode_choices", Value::Bool(false))]).unwrap();
        let d = decode_one(&mut raw, &frame);
        assert_eq!(d.field("message_type"), Some(&Value::Int(0)));
        assert_eq!(d.field("return_code"), Some(&Value::Int(0)));
        let Some(Value::Map(origin)) = d.field("origin") else {
            panic!("{d:?}")
        };
        assert_eq!(origin[3], ("quality".into(), Value::Int(2)));
        assert_eq!(encode(&mut raw, &d), frame);
    }

    #[test]
    fn session_ids_count_per_method() {
        let mut c = open(&[("client_id", Value::Str("0x10".into()))]).unwrap();
        let session = |b: &[u8]| u16::from_be_bytes([b[10], b[11]]);
        let a1 = encode(&mut c, &set_route());
        let a2 = encode(&mut c, &set_route());
        let b1 = encode(&mut c, &status());
        assert_eq!((session(&a1), session(&a2), session(&b1)), (1, 2, 1));
        assert_eq!(&a1[8..10], [0, 0x10]);
        // Given values win and leave the counter alone.
        let mut m = set_route();
        m.fields.push(("session_id".into(), Value::Int(0x4242)));
        m.fields.push(("client_id".into(), Value::Int(3)));
        let given = encode(&mut c, &m);
        assert_eq!(&given[8..12], [0, 3, 0x42, 0x42]);
        assert_eq!(session(&encode(&mut c, &set_route())), 3);
        // The counter skips 0 when it wraps.
        c.sessions.insert((0x1234, 1), u16::MAX);
        assert_eq!(session(&encode(&mut c, &set_route())), 1);
    }

    #[test]
    fn tcp_stream_split_across_chunks() {
        let mut c = codec();
        let messages = [set_route(), status(), set_route_response()];
        let stream: Vec<u8> = messages.iter().flat_map(|m| encode(&mut c, m)).collect();
        let other = encode(&mut c, &status());
        let mut out = Vec::new();
        // Two connections interleaved, in chunks that split headers and payloads.
        let mut other_chunks = other.chunks(7);
        for chunk in stream.chunks(5) {
            let mut p = Packet::bytes(chunk.to_vec());
            p.source = Some(1);
            out.extend(c.decode(p).unwrap());
            if let Some(o) = other_chunks.next() {
                let mut p = Packet::bytes(o.to_vec());
                p.source = Some(2);
                out.extend(c.decode(p).unwrap());
            }
        }
        let names: Vec<_> = out
            .iter()
            .map(|m| (m.name.clone().unwrap(), m.packet.as_ref().unwrap().source))
            .collect();
        assert_eq!(
            names,
            [
                ("VehicleService.Status".to_owned(), Some(2)),
                ("VehicleService.SetRoute".to_owned(), Some(1)),
                ("VehicleService.Status".to_owned(), Some(1)),
                ("VehicleService.SetRoute".to_owned(), Some(1)),
            ]
        );
        assert_eq!(out[3].field("message_type"), Some(&s("RESPONSE")));
        assert!(c.buffers.is_empty());
    }

    #[test]
    fn several_messages_in_one_datagram() {
        let mut c = codec();
        let mut datagram = encode(&mut c, &status());
        datagram.extend(encode(&mut c, &set_route()));
        // An unknown service (and SOME/IP-SD) decodes as a nameless message.
        let unknown = [
            0xFF, 0xFF, 0x81, 0x00, 0, 0, 0, 12, 0, 0, 0, 1, 1, 1, 2, 0, 1, 2, 3, 4,
        ];
        datagram.extend_from_slice(&unknown);
        let out = c.decode(Packet::bytes(datagram)).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].name.as_deref(), Some("VehicleService.Status"));
        assert_eq!(out[1].name.as_deref(), Some("VehicleService.SetRoute"));
        assert_eq!(out[2].name, None);
        assert_eq!(out[2].packet.as_ref().unwrap().data, unknown);
        // So does a known method of another interface version, and SOME/IP-TP.
        let mut other_version = encode(&mut c, &status());
        other_version[13] = 2;
        assert_eq!(decode_one(&mut c, &other_version).name, None);
        let mut tp = encode(&mut c, &status());
        tp[14] |= TP_FLAG;
        assert_eq!(decode_one(&mut c, &tp).name, None);
    }

    #[test]
    fn notifications_have_client_id_zero() {
        let mut c = open(&[("client_id", Value::Int(0x10))]).unwrap();
        assert_eq!(&encode(&mut c, &set_route())[8..10], [0, 0x10]);
        // PRS_SOMEIP_00925: an event comes from the server.
        assert_eq!(&encode(&mut c, &status())[8..10], [0, 0]);
        // Unless the message says otherwise.
        let mut m = status();
        m.fields.push(("client_id".into(), Value::Int(7)));
        assert_eq!(&encode(&mut c, &m)[8..10], [0, 7]);
    }

    #[test]
    fn a_truncated_datagram_is_not_glued_to_the_next() {
        let mut c = codec();
        let mut bad = encode(&mut c, &status());
        bad.extend_from_slice(&encode(&mut c, &set_route())[..20]);
        let e = c.decode(Packet::bytes(bad)).unwrap_err();
        assert!(
            e.to_string().contains("datagram ends inside a message"),
            "{e}"
        );
        assert!(c.buffers.is_empty());
        // The status decoded before the remainder is delivered with the
        // next datagram, which decodes on its own.
        let next = encode(&mut c, &status());
        let out = c.decode(Packet::bytes(next)).unwrap();
        let names: Vec<_> = out.iter().map(|m| m.name.as_deref()).collect();
        assert_eq!(names, [Some("VehicleService.Status"); 2]);

        // A byte stream without a source (a file replay) with stream=True.
        let mut s = open(&[("stream", Value::Bool(true))]).unwrap();
        let bytes = encode(&mut s, &set_route());
        let mut out = Vec::new();
        for chunk in bytes.chunks(9) {
            out.extend(s.decode(Packet::bytes(chunk.to_vec())).unwrap());
        }
        assert_eq!(out.len(), 1);
        assert!(open(&[("stream", Value::Str("maybe".into()))]).is_err());
    }

    #[test]
    fn dynamic_arrays_respect_their_minimum_size() {
        let mut c = codec();
        let mut m = set_route();
        for (k, v) in &mut m.fields {
            if k == "waypoints" {
                *v = Value::List(vec![]);
            }
        }
        let e = c.encode(&m).unwrap_err().to_string();
        assert!(e.contains("fewer than the minimum 1"), "{e}");
    }

    #[test]
    fn unsigned_64_bit_values_keep_their_precision() {
        let mut c = codec();
        for t in [u64::MAX, (1u64 << 63) + 1, 1_700_000_000_123] {
            let mut m = status();
            for (k, v) in &mut m.fields {
                if k == "timestamp" {
                    *v = if let Ok(i) = i64::try_from(t) {
                        Value::Int(i)
                    } else {
                        Value::UInt(t)
                    };
                }
            }
            let bytes = encode(&mut c, &m);
            let d = decode_one(&mut c, &bytes);
            assert_eq!(
                d.field("timestamp").and_then(|v| match v {
                    Value::Int(i) => u64::try_from(*i).ok(),
                    Value::UInt(u) => Some(*u),
                    _ => None,
                }),
                Some(t)
            );
            assert_eq!(encode(&mut c, &d)[16..], bytes[16..]);
        }
    }

    #[test]
    fn error_messages_carry_their_payload() {
        let mut c = codec();
        let m = Message::new(
            "VehicleService.SetRoute",
            map(&[
                ("message_type", s("ERROR")),
                ("return_code", s("E_NOT_READY")),
                ("payload", Value::Array(Array::I64(vec![1, 2]))),
            ]),
        );
        let bytes = encode(&mut c, &m);
        assert_eq!(&bytes[14..], [0x81, 0x04, 1, 2]);
        let d = decode_one(&mut c, &bytes);
        assert_eq!(d.field("return_code"), Some(&s("E_NOT_READY")));
        assert_eq!(
            d.field("payload"),
            Some(&Value::Array(Array::U8(vec![1, 2])))
        );
        assert_eq!(encode(&mut c, &d), bytes);
        let mut unknown_code = m.clone();
        unknown_code.fields[1].1 = Value::Int(0x30);
        unknown_code.fields.pop();
        let bytes = encode(&mut c, &unknown_code);
        let d = decode_one(&mut c, &bytes);
        assert_eq!(d.field("return_code"), Some(&Value::Int(0x30)));
        assert_eq!(d.field("payload"), None);
    }

    #[test]
    fn truncated_payloads_fail() {
        let mut c = codec();
        let good = encode(&mut c, &set_route());
        // Cut the name string short but keep the header consistent.
        let mut cut = good[..good.len() - 3].to_vec();
        let len = (cut.len() - 8) as u32;
        cut[4..8].copy_from_slice(&len.to_be_bytes());
        let after = encode(&mut c, &status());
        let mut datagram = cut.clone();
        datagram.extend_from_slice(&after);
        let e = c.decode(Packet::bytes(datagram)).unwrap_err().to_string();
        assert!(
            e.contains("VehicleService.SetRoute.name") && e.contains("truncated"),
            "{e}"
        );
        // The message after it is decoded too, and arrives with the next
        // packet (this one returned the error).
        let next = encode(&mut c, &status());
        let out = c.decode(Packet::bytes(next)).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out
            .iter()
            .all(|m| m.name.as_deref() == Some("VehicleService.Status")));

        // A struct cut in the middle.
        let mut short = good[..20].to_vec();
        short[4..8].copy_from_slice(&12u32.to_be_bytes());
        let e = c.decode(Packet::bytes(short)).unwrap_err().to_string();
        assert!(e.contains("origin.latitude"), "{e}");
        // A length field below the header's 8 bytes drops the buffer.
        let e = c
            .decode(Packet::bytes(vec![0, 0, 0, 0, 0, 0, 0, 4, 9, 9]))
            .unwrap_err()
            .to_string();
        assert!(e.contains("length field 4"), "{e}");
        assert!(c.buffers.is_empty());
    }

    #[test]
    fn encode_errors() {
        let mut c = codec();
        let err = |c: &mut Someip, m: &Message| c.encode(m).unwrap_err().to_string();
        let e = err(&mut c, &Message::new("VehicleService.Nope", vec![]));
        assert!(
            e.contains("\"VehicleService.Nope\"") && e.contains("VehicleService.SetRoute"),
            "{e}"
        );
        let mut m = set_route();
        m.fields.remove(2);
        assert!(err(&mut c, &m).contains("missing parameter \"name\""));
        let mut m = set_route();
        m.fields.push(("typo".into(), Value::Int(1)));
        assert!(err(&mut c, &m).contains("has no parameter \"typo\""));
        // Response parameters do not belong in a request.
        let mut m = set_route();
        m.fields.push(("accepted".into(), Value::Int(1)));
        assert!(err(&mut c, &m).contains("(REQUEST) has no parameter \"accepted\""));
        let mut m = set_route();
        m.fields[1].1 = Value::List(vec![waypoint(1, 40_000)]);
        let e = err(&mut c, &m);
        assert!(
            e.contains("waypoints[0].speed") && e.contains("out of range"),
            "{e}"
        );
        let mut m = set_route();
        m.fields[1].1 = Value::List((0..9).map(|i| waypoint(i, 0)).collect());
        assert!(err(&mut c, &m).contains("more than the maximum 8"));
        let mut m = set_route();
        m.fields[0].1 = Value::Map(map(&[("latitude", Value::Float(1.0))]));
        assert!(err(&mut c, &m).contains("missing member \"longitude\""));
        let mut m = set_route();
        m.fields[0].1 = position(0.0, 0.0, 0.0, "FOO");
        assert!(err(&mut c, &m).contains("\"FOO\" is not one of NONE, GNSS, RTK_FIXED"));
        let mut m = set_route_response();
        m.fields[3].1 = Value::Array(Array::U32(vec![1, 2]));
        assert!(err(&mut c, &m).contains("has 3 elements, not 2"));
        let mut m = configure("A");
        m.fields[1].1 = Value::Map(vec![]);
        assert!(err(&mut c, &m).contains("exactly one member"));
        let mut m = set_route();
        m.fields.push(("message_type".into(), s("BOGUS")));
        assert!(err(&mut c, &m).contains("REQUEST_NO_RETURN"));
        let mut m = set_route();
        m.fields.push(("method_id".into(), Value::Int(2)));
        assert!(err(&mut c, &m).contains("not the FIBEX's 0x0001"));
        let mut m = set_route();
        m.fields.push(("interface_version".into(), Value::Int(2)));
        assert!(err(&mut c, &m).contains("major version 1"));
    }

    #[test]
    fn option_errors() {
        let err = |o: &[(&str, Value)]| open(o).err().unwrap().to_string();
        let e = err(&[("typo", Value::Int(1))]);
        assert!(
            e.contains("the someip codec does not take") && e.contains("typo"),
            "{e}"
        );
        assert!(err(&[("client_id", Value::Int(70_000))]).contains("16-bit"));
        assert!(err(&[("client_id", s("x"))]).contains("16-bit"));
        assert!(err(&[("decode_choices", s("maybe"))]).contains("not a boolean"));
        let no_file = Someip::open(CodecSpec::parse("someip", Options::new()).unwrap());
        assert!(no_file.err().unwrap().to_string().contains("needs a file"));
        let missing =
            Someip::open(CodecSpec::parse("someip:/nonexistent.xml", Options::new()).unwrap());
        assert!(missing.err().unwrap().to_string().contains("FIBEX error"));
        let network = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/network.xml");
        let no_services =
            Someip::open(CodecSpec::parse(&format!("someip:{network}"), Options::new()).unwrap());
        assert!(no_services
            .err()
            .unwrap()
            .to_string()
            .contains("no SOME/IP service"));
        assert!(crate::codec::names().contains(&"someip".to_string()));
        assert!(open(&[("client_id", Value::Int(0xFFFF))]).is_ok());
    }
}
