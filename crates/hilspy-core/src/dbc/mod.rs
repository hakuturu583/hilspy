//! DBC databases: message and signal definitions parsed with the `can-dbc`
//! crate, and the encoding of physical signal values to payloads and back.

pub(crate) mod bits;
pub mod coding;

use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use can_dbc::{
    AttributeDefinition, AttributeValue, AttributeValueType, Dbc, MessageId, MultiplexIndicator,
    NumericValue, SignalExtendedValueType, ValueType,
};

pub use coding::{Coding, Conversion, Scale};

use crate::error::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ByteOrder {
    LittleEndian,
    BigEndian,
}

/// How the raw bits of a signal are interpreted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueKind {
    /// Unsigned, or two's complement when [`Signal::signed`] (the DBC kinds).
    Integer,
    /// Sign and magnitude: the top bit is the sign, the rest the magnitude.
    SignMagnitude,
    /// One's complement: a negative value is the bitwise NOT of its magnitude.
    OnesComplement,
    Float32,
    Float64,
}

impl ValueKind {
    /// Whether raw values are integers (not IEEE floats).
    pub fn is_integer(self) -> bool {
        matches!(
            self,
            ValueKind::Integer | ValueKind::SignMagnitude | ValueKind::OnesComplement
        )
    }
}

/// A physical signal value: what `encode` takes and `decode` returns.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    /// A value-table entry (`VAL_`) name.
    Str(String),
}

impl Value {
    fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Int(i) => Some(*i as f64),
            Value::Float(f) => Some(*f),
            Value::Str(_) => None,
        }
    }
}

/// The multiplexor a multiplexed signal depends on.
#[derive(Clone, Debug, PartialEq)]
pub struct Multiplexing {
    /// Index of the multiplexor signal in [`Message::signals`].
    pub multiplexor: usize,
    /// Inclusive raw multiplexor value ranges for which the signal is present.
    pub ranges: Vec<(u64, u64)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Signal {
    pub name: String,
    pub start_bit: u32,
    pub size: u32,
    pub byte_order: ByteOrder,
    pub signed: bool,
    pub kind: ValueKind,
    pub factor: f64,
    pub offset: f64,
    /// Physical minimum (`None` when the DBC gives `[0|0]`).
    pub minimum: Option<f64>,
    pub maximum: Option<f64>,
    pub unit: String,
    pub receivers: Vec<String>,
    pub comment: Option<String>,
    /// Value table (raw value, name).
    pub choices: Vec<(i64, String)>,
    /// Raw value of `GenSigStartValue`.
    pub initial_raw: Option<f64>,
    /// Other signals of the message are multiplexed by this one.
    pub is_multiplexor: bool,
    pub multiplexing: Option<Multiplexing>,
    /// A conversion that `factor`, `offset` and `choices` cannot express
    /// (FIBEX compu methods); when set it replaces them. `None` for DBC
    /// signals.
    pub coding: Option<Coding>,
}

impl Signal {
    fn raw_bounds(&self) -> (i128, i128) {
        if matches!(
            self.kind,
            ValueKind::SignMagnitude | ValueKind::OnesComplement
        ) {
            // Symmetric: both have a negative zero instead of a minimum.
            let max = (1i128 << (self.size - 1)) - 1;
            (-max, max)
        } else if self.signed {
            (-(1i128 << (self.size - 1)), (1i128 << (self.size - 1)) - 1)
        } else {
            (0, (1i128 << self.size) - 1)
        }
    }

    /// The smallest and largest raw values.
    pub fn raw_range(&self) -> (f64, f64) {
        match self.kind {
            ValueKind::Float32 => (f64::from(f32::MIN), f64::from(f32::MAX)),
            ValueKind::Float64 => (f64::MIN, f64::MAX),
            _ => {
                let (lo, hi) = self.raw_bounds();
                (lo as f64, hi as f64)
            }
        }
    }

    /// Whether integer raw values scale to integer physical values.
    fn integral_scaling(&self) -> bool {
        self.kind.is_integer() && self.factor.fract() == 0.0 && self.offset.fract() == 0.0
    }

    /// The raw value of a name in the value table.
    pub fn choice_raw(&self, name: &str) -> Option<i64> {
        self.choices
            .iter()
            .find(|(_, n)| n == name)
            .map(|(v, _)| *v)
    }

    fn choice_name(&self, raw: i64) -> Option<&str> {
        self.choices
            .iter()
            .find(|(v, _)| *v == raw)
            .map(|(_, n)| n.as_str())
    }

    /// Interpret the low `size` bits of `bits` as this signal's raw value.
    fn raw_from_bits(&self, bits: u64) -> RawValue {
        match self.kind {
            ValueKind::Float32 => RawValue::Float(f64::from(f32::from_bits(bits as u32))),
            ValueKind::Float64 => RawValue::Float(f64::from_bits(bits)),
            ValueKind::SignMagnitude | ValueKind::OnesComplement => {
                let sign = 1u64 << (self.size - 1);
                let negative = bits & sign != 0;
                let magnitude = if negative && self.kind == ValueKind::OnesComplement {
                    !bits & (sign - 1)
                } else {
                    bits & (sign - 1)
                };
                // Negative zero decodes to 0.
                let magnitude = i128::from(magnitude);
                RawValue::Int(if negative { -magnitude } else { magnitude })
            }
            ValueKind::Integer if self.signed && self.size < 64 => {
                let shift = 64 - self.size;
                RawValue::Int(((bits << shift) as i64 >> shift) as i128)
            }
            ValueKind::Integer if self.signed => RawValue::Int(bits as i64 as i128),
            ValueKind::Integer => RawValue::Int(bits as i128),
        }
    }

    /// Physical value of a raw value.
    fn decode_raw(&self, raw: RawValue, options: &DecodeOptions) -> Value {
        if let Some(coding) = &self.coding {
            return decode_coded(coding, raw, options);
        }
        match raw {
            RawValue::Float(r) => Value::Float(if options.scaling {
                r * self.factor + self.offset
            } else {
                r
            }),
            RawValue::Int(r) => {
                if options.decode_choices {
                    if let Some(name) = i64::try_from(r).ok().and_then(|r| self.choice_name(r)) {
                        return Value::Str(name.to_owned());
                    }
                }
                if !options.scaling {
                    return int_or_float(r);
                }
                if self.integral_scaling() {
                    let phys = r * self.factor as i128 + self.offset as i128;
                    int_or_float(phys)
                } else {
                    Value::Float(r as f64 * self.factor + self.offset)
                }
            }
        }
    }

    /// Bits of the raw value for a physical value.
    fn encode_value(&self, value: &Value, options: &EncodeOptions) -> Result<u64> {
        if let Some(coding) = &self.coding {
            return self.encode_coded(coding, value, options);
        }
        let raw = match value {
            Value::Str(name) => match self.choice_raw(name) {
                Some(raw) => RawValue::Int(i128::from(raw)),
                None => {
                    return Err(Error::encode(format!(
                        "{:?} is not a value of signal {} (choices: {})",
                        name,
                        self.name,
                        self.choices
                            .iter()
                            .map(|(_, n)| n.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )))
                }
            },
            _ => {
                let phys = value.as_f64().unwrap_or_default();
                if !phys.is_finite() && self.kind.is_integer() {
                    return Err(Error::encode(format!(
                        "signal {} cannot hold {phys}",
                        self.name
                    )));
                }
                if options.strict {
                    // The range is physical; a raw value is scaled to compare.
                    let physical = if options.scaling {
                        phys
                    } else {
                        phys * self.factor + self.offset
                    };
                    self.check_range(physical)?;
                }
                if !options.scaling {
                    match value {
                        Value::Int(i) if self.kind.is_integer() => RawValue::Int(i128::from(*i)),
                        _ if self.kind.is_integer() => RawValue::Int(round_raw(phys)),
                        _ => RawValue::Float(phys),
                    }
                } else if !self.kind.is_integer() {
                    RawValue::Float((phys - self.offset) / self.factor)
                } else if let (Value::Int(i), true) = (value, self.integral_scaling()) {
                    // Exact integer arithmetic (64-bit raw values exceed f64).
                    let num = i128::from(*i) - self.offset as i128;
                    let factor = self.factor as i128;
                    if factor == 0 || num % factor != 0 {
                        RawValue::Int(round_raw((phys - self.offset) / self.factor))
                    } else {
                        RawValue::Int(num / factor)
                    }
                } else {
                    RawValue::Int(round_raw((phys - self.offset) / self.factor))
                }
            }
        };
        self.raw_to_bits(raw)
    }

    /// [`Signal::encode_value`] through the signal's coding.
    fn encode_coded(&self, coding: &Coding, value: &Value, options: &EncodeOptions) -> Result<u64> {
        let raw = match value {
            Value::Str(name) => coding.raw_of_name(name).ok_or_else(|| {
                Error::encode(format!(
                    "{:?} is not a value of signal {} (choices: {})",
                    name,
                    self.name,
                    coding.names().collect::<Vec<_>>().join(", ")
                ))
            })?,
            _ => {
                let v = value.as_f64().unwrap_or_default();
                if !v.is_finite() && self.kind.is_integer() {
                    return Err(Error::encode(format!(
                        "signal {} cannot hold {v}",
                        self.name
                    )));
                }
                if !options.scaling {
                    if options.strict {
                        if let Some(physical) = coding.physical(RawValue::Float(v)) {
                            self.check_range(physical)?;
                        }
                    }
                    match value {
                        Value::Int(i) if self.kind.is_integer() => {
                            return self.raw_to_bits(RawValue::Int(i128::from(*i)))
                        }
                        _ => v,
                    }
                } else {
                    if options.strict {
                        self.check_range(v)?;
                    }
                    coding
                        .raw_of_physical(v)
                        .map_err(|e| Error::encode(format!("signal {}: {e}", self.name)))?
                }
            }
        };
        self.raw_to_bits(RawValue::Float(raw))
    }

    fn raw_to_bits(&self, raw: RawValue) -> Result<u64> {
        match (self.kind, raw) {
            (ValueKind::Float32, RawValue::Float(r)) => Ok(u64::from((r as f32).to_bits())),
            (ValueKind::Float32, RawValue::Int(r)) => Ok(u64::from((r as f32).to_bits())),
            (ValueKind::Float64, RawValue::Float(r)) => Ok(r.to_bits()),
            (ValueKind::Float64, RawValue::Int(r)) => Ok((r as f64).to_bits()),
            (_, RawValue::Float(r)) => self.raw_to_bits(RawValue::Int(round_raw(r))),
            (kind, RawValue::Int(r)) => {
                let (lo, hi) = self.raw_bounds();
                if r < lo || r > hi {
                    return Err(Error::encode(format!(
                        "signal {}: raw value {r} does not fit in {} {} bits ({lo}..={hi})",
                        self.name,
                        self.size,
                        match kind {
                            ValueKind::SignMagnitude => "sign-magnitude",
                            ValueKind::OnesComplement => "one's complement",
                            _ if self.signed => "signed",
                            _ => "unsigned",
                        }
                    )));
                }
                let mask = if self.size == 64 {
                    u64::MAX
                } else {
                    (1u64 << self.size) - 1
                };
                let sign = 1u64 << (self.size - 1);
                let magnitude = r.unsigned_abs() as u64;
                Ok(match kind {
                    ValueKind::SignMagnitude if r < 0 => sign | magnitude,
                    ValueKind::OnesComplement if r < 0 => !magnitude & mask,
                    _ => (r as i64 as u64) & mask,
                })
            }
        }
    }

    fn check_range(&self, phys: f64) -> Result<()> {
        // Tolerate the rounding of a range given in decimal.
        let eps = self.factor.abs() * 1e-6;
        if let Some(min) = self.minimum {
            if phys < min - eps {
                return Err(Error::encode(format!(
                    "signal {}: {phys} is below the minimum {min}",
                    self.name
                )));
            }
        }
        if let Some(max) = self.maximum {
            if phys > max + eps {
                return Err(Error::encode(format!(
                    "signal {}: {phys} is above the maximum {max}",
                    self.name
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum RawValue {
    Int(i128),
    Float(f64),
}

/// [`Signal::decode_raw`] through a coding: a name, else the physical value
/// of the first rational scale holding the raw value, else the raw value.
fn decode_coded(coding: &Coding, raw: RawValue, options: &DecodeOptions) -> Value {
    if options.decode_choices {
        if let Some(name) = coding.name(raw) {
            return Value::Str(name.to_owned());
        }
    }
    match (options.scaling.then(|| coding.physical(raw)).flatten(), raw) {
        (Some(physical), _) => Value::Float(physical),
        (None, RawValue::Int(i)) => int_or_float(i),
        (None, RawValue::Float(f)) => Value::Float(f),
    }
}

fn round_raw(r: f64) -> i128 {
    // Saturating; out-of-range values are reported by raw_to_bits.
    r.round() as i128
}

fn int_or_float(v: i128) -> Value {
    i64::try_from(v).map_or(Value::Float(v as f64), Value::Int)
}

#[derive(Clone, Debug, PartialEq)]
pub struct Message {
    pub id: u32,
    pub extended: bool,
    pub name: String,
    /// Payload length in bytes.
    pub size: usize,
    /// The ECUs that send it: the `BO_` transmitter and any `BO_TX_BU_`
    /// ones (a FIBEX frame triggering's output ports).
    pub transmitters: Vec<String>,
    /// The ECUs that receive it: every receiver of its signals (a FIBEX
    /// frame triggering's input ports).
    pub receivers: Vec<String>,
    pub signals: Vec<Signal>,
    pub comment: Option<String>,
    /// How often the message is sent (`GenMsgCycleTime`; a FIBEX frame
    /// triggering's `CYCLIC-TIMING`).
    pub cycle_time: Option<Duration>,
    /// Sent as CAN FD (`VFrameFormat`, or a payload over 8 bytes).
    pub fd: bool,
}

/// Options of [`Message::encode`].
#[derive(Clone, Copy, Debug)]
pub struct EncodeOptions {
    /// Values are physical (scaled); `false` takes raw values.
    pub scaling: bool,
    /// Reject values outside the signal's `[min|max]`, missing signals and
    /// signals that the multiplexor value leaves out. Without it, missing
    /// signals take `GenSigStartValue` (or 0) and absent ones are ignored.
    pub strict: bool,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            scaling: true,
            strict: true,
        }
    }
}

/// Options of [`Message::decode`].
#[derive(Clone, Copy, Debug)]
pub struct DecodeOptions {
    /// Return value-table names instead of numbers where one matches.
    pub decode_choices: bool,
    /// Return physical (scaled) values; `false` returns raw values.
    pub scaling: bool,
    /// Skip signals that lie beyond a short payload instead of failing.
    pub allow_truncated: bool,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            decode_choices: true,
            scaling: true,
            allow_truncated: false,
        }
    }
}

impl Message {
    pub fn signal(&self, name: &str) -> Option<&Signal> {
        self.signals.iter().find(|s| s.name == name)
    }

    fn signal_index(&self, name: &str) -> Option<usize> {
        self.signals.iter().position(|s| s.name == name)
    }

    /// Encode signal values into a payload of [`Message::size`] bytes.
    pub fn encode(&self, values: &[(String, Value)], options: &EncodeOptions) -> Result<Vec<u8>> {
        let mut by_index: Vec<Option<&Value>> = vec![None; self.signals.len()];
        for (name, value) in values {
            let i = self.signal_index(name).ok_or_else(|| {
                Error::encode(format!("message {} has no signal {name:?}", self.name))
            })?;
            by_index[i] = Some(value);
        }
        let mut bits: Vec<Option<u64>> = vec![None; self.signals.len()];
        let mut data = vec![0u8; self.size];
        for i in 0..self.signals.len() {
            let active = self.encode_active(i, &by_index, &mut bits, options, 0)?;
            let signal = &self.signals[i];
            if !active {
                if by_index[i].is_some() && options.strict {
                    return Err(Error::encode(format!(
                        "signal {} of {} is not present for the given multiplexor value",
                        signal.name, self.name
                    )));
                }
                continue;
            }
            let raw = self.signal_bits(i, &by_index, &mut bits, options)?;
            bits::write(
                &mut data,
                signal.start_bit,
                signal.size,
                signal.byte_order,
                raw,
            );
        }
        Ok(data)
    }

    /// The bits of signal `i` from the given values (or its default).
    fn signal_bits(
        &self,
        i: usize,
        values: &[Option<&Value>],
        bits: &mut [Option<u64>],
        options: &EncodeOptions,
    ) -> Result<u64> {
        if let Some(b) = bits[i] {
            return Ok(b);
        }
        let signal = &self.signals[i];
        let b = match values[i] {
            Some(value) => signal.encode_value(value, options)?,
            None if options.strict => {
                return Err(Error::encode(format!(
                "missing value for signal {} of {} (pass strict=False to use its initial value)",
                signal.name, self.name
            )))
            }
            None => match signal.initial_raw {
                Some(raw) => signal.raw_to_bits(RawValue::Float(raw))?,
                None => 0,
            },
        };
        bits[i] = Some(b);
        Ok(b)
    }

    fn encode_active(
        &self,
        i: usize,
        values: &[Option<&Value>],
        bits: &mut [Option<u64>],
        options: &EncodeOptions,
        depth: usize,
    ) -> Result<bool> {
        let Some(mux) = &self.signals[i].multiplexing else {
            return Ok(true);
        };
        if depth > self.signals.len() {
            return Err(Error::encode("cyclic multiplexing"));
        }
        if !self.encode_active(mux.multiplexor, values, bits, options, depth + 1)? {
            return Ok(false);
        }
        let selector = self.signal_bits(mux.multiplexor, values, bits, options)?;
        Ok(in_ranges(&mux.ranges, selector))
    }

    /// Decode the signals present in `data`, in definition order.
    pub fn decode(&self, data: &[u8], options: &DecodeOptions) -> Result<Vec<(String, Value)>> {
        let raw_bits: Vec<Option<u64>> = self
            .signals
            .iter()
            .map(|s| bits::read(data, s.start_bit, s.size, s.byte_order))
            .collect();
        let mut out = Vec::new();
        for (i, signal) in self.signals.iter().enumerate() {
            match self.decode_active(i, &raw_bits, 0) {
                Some(true) => {}
                Some(false) => continue,
                None if options.allow_truncated => continue,
                None => {
                    return Err(Error::decode(format!(
                        "{} bytes is too short for message {} ({} bytes)",
                        data.len(),
                        self.name,
                        self.size
                    )))
                }
            }
            let Some(b) = raw_bits[i] else {
                if options.allow_truncated {
                    continue;
                }
                return Err(Error::decode(format!(
                    "{} bytes is too short for signal {} of {} ({} bytes)",
                    data.len(),
                    signal.name,
                    self.name,
                    self.size
                )));
            };
            out.push((
                signal.name.clone(),
                signal.decode_raw(signal.raw_from_bits(b), options),
            ));
        }
        Ok(out)
    }

    /// `None` if a multiplexor is beyond the payload.
    fn decode_active(&self, i: usize, raw_bits: &[Option<u64>], depth: usize) -> Option<bool> {
        let Some(mux) = &self.signals[i].multiplexing else {
            return Some(true);
        };
        if depth > self.signals.len()
            || !self.decode_active(mux.multiplexor, raw_bits, depth + 1)?
        {
            return Some(false);
        }
        Some(in_ranges(&mux.ranges, raw_bits[mux.multiplexor]?))
    }
}

fn in_ranges(ranges: &[(u64, u64)], v: u64) -> bool {
    ranges.iter().any(|&(lo, hi)| lo <= v && v <= hi)
}

/// A parsed DBC file.
#[derive(Clone, Debug)]
pub struct Database {
    pub version: String,
    pub nodes: Vec<String>,
    pub messages: Vec<Message>,
    by_name: HashMap<String, usize>,
    by_id: HashMap<(u32, bool), usize>,
}

impl Database {
    /// Load a DBC file (UTF-8, or Windows-1252 as Vector tools write).
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref()).map_err(|e| {
            Error::Io(std::io::Error::new(
                e.kind(),
                format!("{}: {e}", path.as_ref().display()),
            ))
        })?;
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let text = match std::str::from_utf8(bytes) {
            Ok(s) => std::borrow::Cow::Borrowed(s),
            Err(_) => can_dbc::decode_cp1252(bytes).ok_or_else(|| {
                Error::DbcParse("the file is neither UTF-8 nor Windows-1252".into())
            })?,
        };
        Self::parse(&text)
    }

    pub fn parse(text: &str) -> Result<Self> {
        let dbc = Dbc::try_from(text).map_err(|e| Error::DbcParse(e.to_string()))?;
        Self::from_dbc(&dbc)
    }

    /// Convert a `can-dbc` AST, resolving value tables, comments, attributes
    /// and multiplexing.
    pub fn from_dbc(dbc: &Dbc) -> Result<Self> {
        let fd_formats = enum_values(dbc, "VFrameFormat");
        let mut messages = Vec::with_capacity(dbc.messages.len());
        for m in &dbc.messages {
            let (id, extended) = match m.id {
                MessageId::Standard(id) => (u32::from(id), false),
                MessageId::Extended(id) => (id, true),
            };
            let size = usize::try_from(m.size).unwrap_or(usize::MAX);
            if size > 64 {
                return Err(Error::DbcParse(format!(
                    "message {} is {size} bytes (at most 64)",
                    m.name
                )));
            }
            let fd_by_attr = match dbc.resolved_message_attribute(m.id, "VFrameFormat") {
                Some(AttributeValue::String(s)) => s.contains("FD"),
                Some(v) => attr_u64(v)
                    .and_then(|i| fd_formats.get(i as usize))
                    .is_some_and(|s| s.contains("FD")),
                None => false,
            };
            let mut signals = Vec::with_capacity(m.signals.len());
            for s in &m.signals {
                signals.push(convert_signal(dbc, m, s)?);
            }
            resolve_multiplexing(dbc, m, &mut signals)?;
            let mut transmitters = Vec::new();
            let extra = dbc
                .message_transmitters
                .iter()
                .filter(|t| t.message_id == m.id)
                .flat_map(|t| &t.transmitter);
            for t in m.transmitter.iter().chain(extra) {
                push_unique(&mut transmitters, t);
            }
            let mut receivers = Vec::new();
            for r in signals.iter().flat_map(|s| &s.receivers) {
                push_unique(&mut receivers, r);
            }
            messages.push(Message {
                id,
                extended,
                name: m.name.clone(),
                size,
                transmitters,
                receivers,
                comment: dbc.message_comment(m.id).map(str::to_owned),
                cycle_time: dbc
                    .resolved_message_attribute(m.id, "GenMsgCycleTime")
                    .and_then(attr_u64)
                    .filter(|&t| t > 0)
                    .map(Duration::from_millis),
                fd: fd_by_attr || size > 8,
                signals,
            });
        }
        Self::new(
            dbc.version.0.clone(),
            dbc.nodes.iter().map(|n| n.0.clone()).collect(),
            messages,
        )
    }

    /// A database from already-built messages (names and identifiers must be
    /// unique).
    pub fn new(version: String, nodes: Vec<String>, messages: Vec<Message>) -> Result<Self> {
        let mut by_name = HashMap::new();
        let mut by_id = HashMap::new();
        for (i, m) in messages.iter().enumerate() {
            if by_name.insert(m.name.clone(), i).is_some() {
                return Err(Error::DbcParse(format!(
                    "duplicate message name {}",
                    m.name
                )));
            }
            if by_id.insert((m.id, m.extended), i).is_some() {
                return Err(Error::DbcParse(format!(
                    "duplicate message identifier 0x{:X} ({})",
                    m.id, m.name
                )));
            }
        }
        Ok(Self {
            version,
            nodes,
            messages,
            by_name,
            by_id,
        })
    }

    /// The ECUs of the network: its nodes (`BU_`, FIBEX ECUs), then any
    /// other ECU a message names as transmitter or receiver.
    pub fn ecus(&self) -> Vec<&str> {
        let mut out: Vec<&str> = Vec::new();
        let named = self
            .messages
            .iter()
            .flat_map(|m| m.transmitters.iter().chain(&m.receivers));
        for e in self.nodes.iter().chain(named) {
            if !out.contains(&e.as_str()) {
                out.push(e);
            }
        }
        out
    }

    pub fn message_by_name(&self, name: &str) -> Result<&Message> {
        self.by_name
            .get(name)
            .map(|&i| &self.messages[i])
            .ok_or_else(|| Error::NotFound(format!("no message named {name:?}")))
    }

    pub fn message_by_id(&self, id: u32, extended: bool) -> Result<&Message> {
        self.by_id
            .get(&(id, extended))
            .map(|&i| &self.messages[i])
            .ok_or_else(|| {
                Error::NotFound(format!(
                    "no message with {} identifier 0x{id:X}",
                    if extended { "extended" } else { "standard" }
                ))
            })
    }
}

fn push_unique(list: &mut Vec<String>, item: &str) {
    if !list.iter().any(|i| i == item) {
        list.push(item.to_owned());
    }
}

fn attr_u64(v: &AttributeValue) -> Option<u64> {
    match v {
        AttributeValue::Uint(u) => Some(*u),
        AttributeValue::Int(i) => u64::try_from(*i).ok(),
        AttributeValue::Double(d) if *d >= 0.0 => Some(*d as u64),
        _ => None,
    }
}

fn attr_f64(v: &AttributeValue) -> Option<f64> {
    match v {
        AttributeValue::Uint(u) => Some(*u as f64),
        AttributeValue::Int(i) => Some(*i as f64),
        AttributeValue::Double(d) => Some(*d),
        AttributeValue::String(_) => None,
    }
}

/// The value names of an enum attribute definition.
fn enum_values(dbc: &Dbc, name: &str) -> Vec<String> {
    dbc.attribute_definitions
        .iter()
        .find_map(|d| match d {
            AttributeDefinition::Message(n, AttributeValueType::Enum(values))
            | AttributeDefinition::Plain(n, AttributeValueType::Enum(values))
                if n == name =>
            {
                Some(values.clone())
            }
            _ => None,
        })
        .unwrap_or_default()
}

fn numeric(v: &NumericValue) -> f64 {
    match v {
        NumericValue::Uint(u) => *u as f64,
        NumericValue::Int(i) => *i as f64,
        NumericValue::Double(d) => *d,
    }
}

fn convert_signal(dbc: &Dbc, m: &can_dbc::Message, s: &can_dbc::Signal) -> Result<Signal> {
    let bad = |what: String| Error::DbcParse(format!("signal {} of {}: {what}", s.name, m.name));
    let byte_order = match s.byte_order {
        can_dbc::ByteOrder::LittleEndian => ByteOrder::LittleEndian,
        can_dbc::ByteOrder::BigEndian => ByteOrder::BigEndian,
    };
    let kind = match dbc.extended_value_type_for_signal(m.id, &s.name) {
        Some(SignalExtendedValueType::IEEEfloat32Bit) => ValueKind::Float32,
        Some(SignalExtendedValueType::IEEEdouble64bit) => ValueKind::Float64,
        _ => ValueKind::Integer,
    };
    let start_bit = u32::try_from(s.start_bit).map_err(|_| bad("start bit out of range".into()))?;
    let size = u32::try_from(s.size).map_err(|_| bad("size out of range".into()))?;
    match kind {
        ValueKind::Float32 if size != 32 => {
            return Err(bad(format!("a float is 32 bits, not {size}")))
        }
        ValueKind::Float64 if size != 64 => {
            return Err(bad(format!("a double is 64 bits, not {size}")))
        }
        _ => {}
    }
    let len = usize::try_from(m.size).unwrap_or(0);
    if !bits::fits(start_bit, size, byte_order, len) {
        return Err(bad(format!(
            "{size} bits from bit {start_bit} do not fit in {len} bytes"
        )));
    }
    if s.factor == 0.0 || !s.factor.is_finite() {
        return Err(bad(format!("invalid factor {}", s.factor)));
    }
    let (min, max) = (numeric(&s.min), numeric(&s.max));
    let (minimum, maximum) = if min == 0.0 && max == 0.0 {
        (None, None)
    } else {
        (Some(min), Some(max))
    };
    let mut choices: Vec<(i64, String)> = dbc
        .value_descriptions_for_signal(m.id, &s.name)
        .map(|d| d.iter().map(|v| (v.id, v.description.clone())).collect())
        .unwrap_or_default();
    choices.sort_by_key(|(v, _)| *v);
    Ok(Signal {
        name: s.name.clone(),
        start_bit,
        size,
        byte_order,
        signed: s.value_type == ValueType::Signed,
        kind,
        factor: s.factor,
        offset: s.offset,
        minimum,
        maximum,
        unit: s.unit.clone(),
        receivers: s.receivers.clone(),
        comment: dbc.signal_comment(m.id, &s.name).map(str::to_owned),
        choices,
        initial_raw: dbc
            .resolved_signal_attribute(m.id, &s.name, "GenSigStartValue")
            .and_then(attr_f64),
        is_multiplexor: matches!(
            s.multiplexer_indicator,
            MultiplexIndicator::Multiplexor
                | MultiplexIndicator::MultiplexorAndMultiplexedSignal(_)
        ),
        multiplexing: None,
        coding: None,
    })
}

fn resolve_multiplexing(dbc: &Dbc, m: &can_dbc::Message, signals: &mut [Signal]) -> Result<()> {
    let plain_multiplexors: Vec<usize> = m
        .signals
        .iter()
        .enumerate()
        .filter(|(_, s)| s.multiplexer_indicator == MultiplexIndicator::Multiplexor)
        .map(|(i, _)| i)
        .collect();
    let index = |name: &str| m.signals.iter().position(|s| s.name == name);
    for (i, s) in m.signals.iter().enumerate() {
        let selector = match s.multiplexer_indicator {
            MultiplexIndicator::MultiplexedSignal(v)
            | MultiplexIndicator::MultiplexorAndMultiplexedSignal(v) => v,
            _ => continue,
        };
        let extended: Vec<_> = dbc
            .extended_multiplex
            .iter()
            .filter(|e| e.message_id == m.id && e.signal_name == s.name)
            .collect();
        let multiplexing = if let Some(first) = extended.first() {
            let multiplexor = index(&first.multiplexor_signal_name).ok_or_else(|| {
                Error::DbcParse(format!(
                    "signal {} of {}: unknown multiplexor {}",
                    s.name, m.name, first.multiplexor_signal_name
                ))
            })?;
            Multiplexing {
                multiplexor,
                ranges: extended
                    .iter()
                    .flat_map(|e| e.mappings.iter().map(|r| (r.min_value, r.max_value)))
                    .collect(),
            }
        } else {
            match plain_multiplexors.as_slice() {
                [mux] if *mux != i => Multiplexing {
                    multiplexor: *mux,
                    ranges: vec![(selector, selector)],
                },
                _ => {
                    return Err(Error::DbcParse(format!(
                        "signal {} of {} is multiplexed but the message has no single multiplexor \
                         (and no SG_MUL_VAL_ entry)",
                        s.name, m.name
                    )))
                }
            }
        };
        signals[i].multiplexing = Some(multiplexing);
    }
    Ok(())
}

#[cfg(test)]
mod tests;
