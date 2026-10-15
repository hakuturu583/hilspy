//! ASAM FIBEX (MCD-2 NET) network descriptions: the CAN / CAN FD frames of
//! one channel, converted to a DBC [`Database`] so the `dbc` encoding,
//! decoding and E2E protection apply unchanged.
//!
//! The `afibex` crate parses the file: channels, frames with their PDU
//! instances, PDUs with their signal instances, signals and the coded types
//! of codings. What it leaves out is read from the same document with
//! `roxmltree`, by element ID: the frame triggerings of a channel (CAN
//! identifier, extended addressing, CAN FD) and the compu methods of a coding
//! (rational coefficients, `SCALE-LINEAR`, texttable ranges, physical
//! constraints, units).
//!
//! Layout: a PDU instance's `BIT-POSITION` is the PDU's offset in the frame
//! (a whole number of bytes). A signal instance's `BIT-POSITION` is read as a
//! DBC start bit within its PDU (bit 0 is the least significant bit of byte
//! 0): the least significant bit for `IS-HIGH-LOW-BYTE-ORDER` false (Intel,
//! the default), the most significant bit for true (Motorola), as `canmatrix`
//! reads and writes FIBEX.

use std::collections::HashMap;
use std::path::Path;

use afibex::fibex::{BaseDataType, Category, Encoding, FibexData};
use roxmltree::{Document, Node};

use crate::dbc::{bits, ByteOrder, Database, Message, Signal, ValueKind};
use crate::error::{Error, Result};

fn bad(msg: impl Into<String>) -> Error {
    Error::FibexParse(msg.into())
}

/// The frames of one channel of a FIBEX file as a database. `channel` is a
/// channel short name or ID; it may be left out when the file has a single
/// channel.
pub fn load(path: impl AsRef<Path>, channel: Option<&str>) -> Result<Database> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|e| {
        Error::Io(std::io::Error::new(
            e.kind(),
            format!("{}: {e}", path.display()),
        ))
    })?;
    let xml = Document::parse(&text).map_err(|e| bad(format!("{}: {e}", path.display())))?;
    let root = xml.root_element().tag_name().name();
    if root != "FIBEX" {
        return Err(bad(format!(
            "{}: the root element is {root}, not FIBEX",
            path.display()
        )));
    }
    let mut data = FibexData::new();
    data.load_fibex_file(path)
        .map_err(|e| bad(format!("{}: {e}", path.display())))?;
    let ids = xml
        .descendants()
        .filter_map(|n| Some((attr(n, "ID")?, n)))
        .collect();
    Network { data: &data, ids }.database(channel)
}

/// The value of an attribute by local name (any namespace).
fn attr<'a>(n: Node<'a, '_>, name: &str) -> Option<&'a str> {
    n.attributes().find(|a| a.name() == name).map(|a| a.value())
}

fn children<'a, 'i>(n: Node<'a, 'i>, name: &'a str) -> impl Iterator<Item = Node<'a, 'i>> + 'a
where
    'i: 'a,
{
    n.children()
        .filter(move |c| c.is_element() && c.tag_name().name() == name)
}

fn child<'a, 'i>(n: Node<'a, 'i>, name: &'a str) -> Option<Node<'a, 'i>>
where
    'i: 'a,
{
    children(n, name).next()
}

fn text<'a>(n: Node<'a, '_>) -> &'a str {
    n.text().unwrap_or_default().trim()
}

fn number(n: Node<'_, '_>, what: &str) -> Result<f64> {
    text(n)
        .parse()
        .map_err(|_| bad(format!("{what}: {:?} is not a number", text(n))))
}

/// A frame triggering: how a channel carries a frame.
struct Triggering {
    id: u32,
    extended: bool,
    fd: bool,
    frame: String,
}

/// The physical meaning of a coding: what the DBC model expresses.
#[derive(Default)]
struct Compu {
    linear: Option<(f64, f64)>,
    choices: Vec<(i64, String)>,
    minimum: Option<f64>,
    maximum: Option<f64>,
    unit: String,
}

impl Compu {
    fn set_linear(&mut self, coding: &str, factor: f64, offset: f64) -> Result<()> {
        match self.linear {
            Some(l) if l != (factor, offset) => Err(bad(format!(
                "coding {coding}: scales with different coefficients (piecewise linear) are not \
                 supported"
            ))),
            _ => {
                self.linear = Some((factor, offset));
                Ok(())
            }
        }
    }
}

struct Network<'d, 'a, 'i> {
    data: &'d FibexData,
    /// Elements by `ID` attribute.
    ids: HashMap<&'a str, Node<'a, 'i>>,
}

impl Network<'_, '_, '_> {
    fn database(&self, channel: Option<&str>) -> Result<Database> {
        let channel = self.select(channel)?;
        let node = self
            .ids
            .get(channel)
            .copied()
            .ok_or_else(|| bad(format!("channel {channel} not found")))?;
        let mut messages = Vec::new();
        for t in node
            .descendants()
            .filter(|n| n.is_element() && n.tag_name().name() == "FRAME-TRIGGERING")
        {
            if let Some(t) = triggering(t)? {
                messages.push(self.message(&t)?);
            }
        }
        Database::new(String::new(), Vec::new(), messages).map_err(|e| match e {
            Error::DbcParse(m) => bad(format!("channel {channel}: {m}")),
            e => e,
        })
    }

    /// The ID of the selected channel.
    fn select(&self, wanted: Option<&str>) -> Result<&str> {
        let mut channels: Vec<_> = self.data.elements.channels.values().collect();
        channels.sort_by(|a, b| a.id.cmp(&b.id));
        let list = || {
            channels
                .iter()
                .map(|c| match &c.short_name {
                    Some(n) => format!("{n} ({})", c.id),
                    None => c.id.clone(),
                })
                .collect::<Vec<_>>()
                .join(", ")
        };
        match wanted {
            Some(w) => channels
                .iter()
                .find(|c| c.id == w)
                .or_else(|| channels.iter().find(|c| c.short_name.as_deref() == Some(w)))
                .map(|c| c.id.as_str())
                .ok_or_else(|| {
                    Error::Config(format!(
                        "option channel: no FIBEX channel {w:?} (channels: {})",
                        list()
                    ))
                }),
            None => match channels.as_slice() {
                [c] => Ok(c.id.as_str()),
                [] => Err(Error::Config("the FIBEX file has no channels".into())),
                _ => Err(Error::Config(format!(
                    "the FIBEX file has {} channels; choose one with the channel option \
                     (channels: {})",
                    channels.len(),
                    list()
                ))),
            },
        }
    }

    fn message(&self, t: &Triggering) -> Result<Message> {
        let frame = self
            .data
            .elements
            .frames_map_by_id
            .get(&t.frame)
            .ok_or_else(|| {
                bad(format!(
                    "frame triggering refers to unknown frame {}",
                    t.frame
                ))
            })?;
        let name = frame.short_name.clone().unwrap_or_else(|| frame.id.clone());
        let size = frame.byte_length as usize;
        if size > 64 {
            return Err(bad(format!("frame {name} is {size} bytes (at most 64)")));
        }
        // (PDU name, signal name, signal) for every signal of every PDU.
        let mut signals = Vec::new();
        for instance in &frame.pdu_instances {
            let pdu = self
                .data
                .elements
                .pdus_map_by_id
                .get(&instance.pdu_ref)
                .ok_or_else(|| {
                    bad(format!(
                        "frame {name} refers to unknown PDU {}",
                        instance.pdu_ref
                    ))
                })?;
            let pdu_name = pdu
                .short_name
                .clone()
                .unwrap_or_else(|| instance.pdu_ref.clone());
            if pdu.multiplexer.is_some() {
                return Err(bad(format!(
                    "PDU {pdu_name}: multiplexed PDUs are not supported"
                )));
            }
            // A high-low PDU instance would place the PDU's bytes in another
            // order than its signals' positions describe; reject it rather
            // than map signals to the wrong bytes.
            if instance.is_high_low_byte_order == Some(true) {
                return Err(bad(format!(
                    "PDU {pdu_name} in frame {name}: high-low byte order PDU instances are \
                     not supported"
                )));
            }
            let offset = instance.bit_position.unwrap_or(0);
            let len = pdu.byte_length as usize;
            if offset % 8 != 0 || (offset / 8) as usize + len > size {
                return Err(bad(format!(
                    "PDU {pdu_name} ({len} bytes at bit {offset}) does not fit at a byte offset \
                     in frame {name} ({size} bytes)"
                )));
            }
            for s in &pdu.signal_instances {
                let (signal_name, mut signal) = self.signal(&pdu_name, s, len)?;
                signal.start_bit += offset;
                signals.push((pdu_name.clone(), signal_name, signal));
            }
        }
        let mut count: HashMap<&str, Vec<&str>> = HashMap::new();
        for (pdu, signal, _) in &signals {
            count.entry(signal).or_default().push(pdu);
        }
        let clashing: Vec<String> = count
            .into_iter()
            .filter(|(_, pdus)| pdus.iter().any(|p| p != &pdus[0]))
            .map(|(s, _)| s.to_owned())
            .collect();
        let mut out: Vec<Signal> = Vec::with_capacity(signals.len());
        for (pdu, signal_name, mut signal) in signals {
            signal.name = if clashing.contains(&signal_name) {
                format!("{pdu}.{signal_name}")
            } else {
                signal_name
            };
            if out.iter().any(|s| s.name == signal.name) {
                return Err(bad(format!(
                    "frame {name} has the signal {} twice",
                    signal.name
                )));
            }
            out.push(signal);
        }
        Ok(Message {
            id: t.id,
            extended: t.extended,
            name,
            size,
            transmitter: None,
            signals: out,
            comment: frame.desc.clone(),
            cycle_time_ms: None,
            fd: t.fd || size > 8,
        })
    }

    /// A signal instance of a PDU of `len` bytes, positioned in the PDU.
    fn signal(
        &self,
        pdu: &str,
        instance: &afibex::fibex::SignalInstance,
        len: usize,
    ) -> Result<(String, Signal)> {
        let signal = self
            .data
            .elements
            .signals_map_by_id
            .get(&instance.signal_ref)
            .ok_or_else(|| {
                bad(format!(
                    "PDU {pdu} refers to unknown signal {}",
                    instance.signal_ref
                ))
            })?;
        let name = signal
            .short_name
            .clone()
            .unwrap_or_else(|| signal.id.clone());
        let what = format!("signal {name} of PDU {pdu}");
        let coded = self
            .data
            .pi
            .codings
            .get(&signal.coding_ref)
            .and_then(|c| c.coded_type.as_ref())
            .ok_or_else(|| {
                bad(format!(
                    "{what}: coding {} is missing or has no supported CODED-TYPE",
                    signal.coding_ref
                ))
            })?;
        if !matches!(coded.category, Category::StandardLengthType) {
            return Err(bad(format!(
                "{what}: only STANDARD-LENGTH-TYPE codings are supported"
            )));
        }
        let size = coded
            .bit_length
            .ok_or_else(|| bad(format!("{what}: the coding has no BIT-LENGTH")))?;
        let float = matches!(coded.encoding, Some(Encoding::IeeeFloating))
            || matches!(
                coded.base_data_type,
                Some(BaseDataType::AFloat32 | BaseDataType::AFloat64)
            );
        let (kind, signed) = if float {
            match size {
                32 => (ValueKind::Float32, true),
                64 => (ValueKind::Float64, true),
                _ => return Err(bad(format!("{what}: a float is 32 or 64 bits, not {size}"))),
            }
        } else {
            let signed = match &coded.encoding {
                Some(Encoding::Signed | Encoding::E2C) => true,
                Some(Encoding::Unsigned | Encoding::Bit) => false,
                None => matches!(
                    coded.base_data_type,
                    Some(
                        BaseDataType::AInt8
                            | BaseDataType::AInt16
                            | BaseDataType::AInt32
                            | BaseDataType::AInt64
                    )
                ),
                Some(e) => return Err(bad(format!("{what}: unsupported encoding {e:?}"))),
            };
            (ValueKind::Integer, signed)
        };
        let start_bit = instance
            .bit_position
            .ok_or_else(|| bad(format!("{what}: the signal instance has no BIT-POSITION")))?;
        let byte_order = if instance.is_high_low_byte_order.unwrap_or(false) {
            ByteOrder::BigEndian
        } else {
            ByteOrder::LittleEndian
        };
        if !bits::fits(start_bit, size, byte_order, len) {
            return Err(bad(format!(
                "{what}: {size} bits from bit {start_bit} do not fit in {len} bytes"
            )));
        }
        let compu = self.compu(&signal.coding_ref)?;
        let (factor, offset) = compu.linear.unwrap_or((1.0, 0.0));
        Ok((
            name.clone(),
            Signal {
                name,
                start_bit,
                size,
                byte_order,
                signed,
                kind,
                factor,
                offset,
                minimum: compu.minimum,
                maximum: compu.maximum,
                unit: compu.unit,
                receivers: Vec::new(),
                comment: signal.desc.clone(),
                choices: compu.choices,
                initial_raw: None,
                is_multiplexor: false,
                multiplexing: None,
            },
        ))
    }

    /// The compu methods of a coding: IDENTICAL, LINEAR, SCALE-LINEAR and
    /// TEXTTABLE.
    fn compu(&self, coding: &str) -> Result<Compu> {
        let mut c = Compu::default();
        let Some(node) = self.ids.get(coding).copied() else {
            return Err(bad(format!("coding {coding} not found")));
        };
        let methods = child(node, "COMPU-METHODS")
            .into_iter()
            .flat_map(|m| children(m, "COMPU-METHOD"));
        for method in methods {
            let category = child(method, "CATEGORY").map(text).unwrap_or_default();
            let unit = child(method, "UNIT-REF")
                .and_then(|u| attr(u, "ID-REF"))
                .and_then(|id| self.ids.get(id).copied());
            if let Some(unit) = unit {
                let name = child(unit, "DISPLAY-NAME").or_else(|| child(unit, "SHORT-NAME"));
                c.unit = name.map(text).unwrap_or_default().to_owned();
            }
            let constraint = child(method, "PHYS-CONSTRS")
                .into_iter()
                .flat_map(|p| children(p, "SCALE-CONSTR"))
                .find(|s| attr(*s, "VALIDITY").is_none_or(|v| v == "VALID"));
            if let Some(s) = constraint {
                c.minimum = limit(s, "LOWER-LIMIT", coding)?.or(c.minimum);
                c.maximum = limit(s, "UPPER-LIMIT", coding)?.or(c.maximum);
            }
            let scales: Vec<_> = child(method, "COMPU-INTERNAL-TO-PHYS")
                .and_then(|i| child(i, "COMPU-SCALES"))
                .into_iter()
                .flat_map(|s| children(s, "COMPU-SCALE"))
                .collect();
            match category {
                "IDENTICAL" => c.set_linear(coding, 1.0, 0.0)?,
                "LINEAR" | "SCALE-LINEAR" | "TEXTTABLE" => {
                    for scale in scales {
                        let vt = child(scale, "COMPU-CONST").and_then(|k| child(k, "VT"));
                        let coeffs = child(scale, "COMPU-RATIONAL-COEFFS");
                        match (vt, coeffs) {
                            (Some(vt), _) if category != "LINEAR" => {
                                c.choices
                                    .push((choice_value(scale, coding)?, text(vt).into()));
                            }
                            (None, Some(r)) if category != "TEXTTABLE" => {
                                let (factor, offset) = rational(r, coding)?;
                                c.set_linear(coding, factor, offset)?;
                            }
                            _ => {
                                return Err(bad(format!(
                                    "coding {coding}: unsupported {category} scale"
                                )))
                            }
                        }
                    }
                }
                other => {
                    return Err(bad(format!(
                        "coding {coding}: unsupported compu method {other:?} (supported: \
                         IDENTICAL, LINEAR, SCALE-LINEAR, TEXTTABLE)"
                    )))
                }
            }
        }
        c.choices.sort_by_key(|(v, _)| *v);
        Ok(c)
    }
}

/// A scale constraint limit (`None` when absent or infinite).
fn limit(scale: Node<'_, '_>, name: &str, coding: &str) -> Result<Option<f64>> {
    match child(scale, name) {
        Some(l) if attr(l, "INTERVAL-TYPE") != Some("INFINITE") => {
            number(l, &format!("coding {coding} {name}")).map(Some)
        }
        _ => Ok(None),
    }
}

/// The raw value of a texttable entry (a single value: lower = upper limit).
fn choice_value(scale: Node<'_, '_>, coding: &str) -> Result<i64> {
    let lower = child(scale, "LOWER-LIMIT").ok_or_else(|| {
        bad(format!(
            "coding {coding}: a texttable scale has no LOWER-LIMIT"
        ))
    })?;
    let value = text(lower);
    let upper = child(scale, "UPPER-LIMIT").map(text).unwrap_or(value);
    let parse = |s: &str| {
        s.parse::<i64>().ok().or_else(|| {
            s.parse::<f64>()
                .ok()
                .filter(|f| f.fract() == 0.0)
                .map(|f| f as i64)
        })
    };
    match (parse(value), parse(upper)) {
        (Some(lo), Some(hi)) if lo == hi => Ok(lo),
        (Some(_), Some(_)) => Err(bad(format!(
            "coding {coding}: texttable ranges ({value}..{upper}) are not supported"
        ))),
        _ => Err(bad(format!(
            "coding {coding}: texttable limit {value:?} is not an integer"
        ))),
    }
}

/// `(factor, offset)` of `COMPU-RATIONAL-COEFFS` `(n0 + n1 x) / d0`.
fn rational(coeffs: Node<'_, '_>, coding: &str) -> Result<(f64, f64)> {
    let values = |name| -> Result<Vec<f64>> {
        child(coeffs, name)
            .into_iter()
            .flat_map(|n| children(n, "V"))
            .map(|v| number(v, &format!("coding {coding} {name}")))
            .collect()
    };
    let numerator = values("COMPU-NUMERATOR")?;
    let denominator = values("COMPU-DENOMINATOR")?;
    let d = match denominator.as_slice() {
        [] => 1.0,
        [d] => *d,
        _ => {
            return Err(bad(format!(
                "coding {coding}: only linear rational coefficients are supported"
            )))
        }
    };
    match numerator.as_slice() {
        [offset, factor] if d != 0.0 && *factor != 0.0 && (factor / d).is_finite() => {
            Ok((factor / d, offset / d))
        }
        _ => Err(bad(format!(
            "coding {coding}: linear coefficients need two numerator values, a non-zero factor \
             and a non-zero denominator"
        ))),
    }
}

/// The CAN identifier and format of a frame triggering; `None` for one
/// without an identifier (not CAN).
fn triggering(t: Node<'_, '_>) -> Result<Option<Triggering>> {
    let tid = attr(t, "ID").unwrap_or_default();
    let Some(value) = t
        .descendants()
        .find(|n| n.is_element() && n.tag_name().name() == "IDENTIFIER-VALUE")
    else {
        return Ok(None);
    };
    let s = text(value);
    let id = match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        Some(hex) => u32::from_str_radix(hex, 16).ok(),
        None => s.parse().ok(),
    }
    .filter(|&id| id <= 0x1FFF_FFFF)
    .ok_or_else(|| {
        bad(format!(
            "frame triggering {tid}: {s:?} is not a CAN identifier"
        ))
    })?;
    let has = |name: &str, value: &str| {
        t.descendants()
            .any(|n| n.is_element() && n.tag_name().name() == name && text(n) == value)
    };
    let extended = matches!(attr(value, "EXTENDED-ADDRESSING"), Some("true" | "1"))
        || has("CAN-ADDRESSING-MODE", "EXTENDED")
        || id > 0x7FF;
    let fd = has("CAN-FRAME-TX-BEHAVIOR", "CAN-FD") || has("CAN-FRAME-RX-BEHAVIOR", "CAN-FD");
    let frame = child(t, "FRAME-REF")
        .and_then(|f| attr(f, "ID-REF"))
        .ok_or_else(|| bad(format!("frame triggering {tid} has no FRAME-REF")))?;
    Ok(Some(Triggering {
        id,
        extended,
        fd,
        frame: frame.to_owned(),
    }))
}

#[cfg(test)]
mod tests;
